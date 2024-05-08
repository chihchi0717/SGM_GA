# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:11:20 2020

@author: ck101

本檔案函式功能為進行參數組的建模可行性判定，確認能夠建出實體模型後，計算結構輪廓的頂點座標。
draw_()包含:
1.draw_tri(): 判定三角形
2.draw_qua(): 判定四邊形
3.draw_hept(): 判定七邊形
4.draw_double(): 判定複合三角形、複合四邊形
"""

import matplotlib.pyplot as plt
import numpy as np
import math as m
import cv2
import matplotlib.pyplot as plt
import os

def draw_tri(sid,ang):
    '''計算三角形頂點座標'''
    rad = m.pi/180 # 轉換弧度
    v_x = [] # 紀錄 X 軸向量
    v_y = [] # 紀錄 Y 軸向量
    S1 = sid[0] # 垂直邊
    S2 = sid[1] # 主斜邊
    A1 = ang[0] # 垂直邊與主斜邊夾角
    S3 = (S1**2 + S2**2 - S1*S2*2*m.cos(A1*rad))**0.5 # 餘弦定理推導第三邊
    A2 = m.acos((S1**2 + S3**2 - S2**2)/(S1*S3*2))/rad # 餘弦定理推導另一角

    '''尖端到垂直邊下方頂點的向量'''
    if A2 == 90: # 狀況一，直角
        v_x.append(S3)
        v_y.append(0)
    elif A2 == 180: # 狀況二，退化為邊 
        v_x.append(0)
        v_y.append(-1*S3)
    elif A2 <90: # 狀況三，其他角 #modify elif A2 >90 or A2 <90:
        v_x.append(S3*m.sin(A2*rad))
        v_y.append(S3*m.cos(A2*rad))
    elif A2 >90: #modify
        print("(draw_New) A2 angle > 90 !")
        return None, None

    '''垂直邊向量'''
    v_x.append(0)
    v_y.append(S1)
    return v_x, v_y

def draw_qua(sid,ang):
    '''計算四邊形頂點座標'''
    rad = m.pi/180 # 轉換弧度
    v_x = []
    v_y = []
    second_tri = -1
    '''第二個三角形在上 second_tri = 1 在下= -1'''
    if second_tri == 1:
        '''第二個三角形在上'''
        S1 = sid[0]
        S2 = sid[1]
        S3 = sid[2]
        S4 = (S1**2 + S2**2 - S1*S2*2*m.cos(ang[0]*rad))**0.5 # 最下面邊
        A1 = m.acos((S1**2 + S4**2 - sid[1]**2)/(S1*S4*2))/rad
        A2 = ang[0] + ang[1]

        '''第一向量(最下面邊)'''
        if A1 == 90: # 狀況一，直角
            v_x.append(S4)
            v_y.append(0)
        elif A1 == 180: # 狀況二，退化為邊 
            v_x.append(0)
            v_y.append(-1*S4)
        elif A1 >90 or A1 <90: # 狀況三，其他角 
            v_x.append(S4*m.sin(A1*rad))
            v_y.append(S4*m.cos(A1*rad))
            
        '''第二向量 S1'''
        v_x.append(0)
        v_y.append(S1)

        '''第三向量 S2'''
        i_ang = 180 - A2 # 判定角
        if i_ang==90: # 狀況一，直角
            v_x.append(S3)
            v_y.append(0)
        elif i_ang==0: # 狀況二，退化為邊 
            v_x.append(0)
            v_y.append(S3)
        elif i_ang>90 or i_ang<90: # 狀況三，其他角 
            v_x.append(S3*m.sin(A2*rad))
            v_y.append((-1)*S3*m.cos(A2*rad))

    else:
        '''第二個三角形在下'''
        S1 = sid[0]
        S2 = sid[1]
        S3 = sid[2]
        S4 = (S1**2 + S2**2 - S1*S2*2*m.cos(ang[0]*rad))**0.5
        a1 = m.acos((S1**2 + S4**2 - sid[1]**2)/(S1*S4*2))/rad
        A1 = a1 + ang[1]
        A2 = ang[0]

        '''第一向量(最下面邊)'''
        if A1 == 90: # 狀況一，直角
            v_x.append(S3)
            v_y.append(0)
        elif A1 == 180: # 狀況二，退化為邊 
            v_x.append(0)
            v_y.append(-1*S3)
        elif A1 >90 or A1 <90: # 狀況三，其他角 
            v_x.append(S3*m.sin(A1*rad))
            v_y.append(S3*m.cos(A1*rad))
            
        '''第二向量 S1'''
        v_x.append(0)
        v_y.append(S1)

        '''第三向量 S2'''
        i_ang = 180 - A2 # 判定角
        if i_ang==90: # 狀況一，直角
            v_x.append(S2)
            v_y.append(0)
        elif i_ang==0: # 狀況二，退化為邊 
            v_x.append(0)
            v_y.append(S2)
        elif i_ang>90 or i_ang<90: # 狀況三，其他角 
            v_x.append(S2*m.sin(A2*rad))
            v_y.append((-1)*S2*m.cos(A2*rad))
    return v_x, v_y

def draw_hept(sid,ang):
    '''計算七邊形頂點座標'''
    rad = m.pi/180 # 轉換弧度
    v_x = []
    v_y = []
    S1 = sid[0]
    S2 = sid[1]
    S3 = sid[2]
    S4 = sid[3]
    S5 = sid[4]
    S6 = sid[5]
    S1_2 = (S1**2 + S2**2 - S1*S2*2*m.cos(ang[0]*rad))**0.5
    S2_3 = (S2**2 + S3**2 - S2*S3*2*m.cos(ang[1]*rad))**0.5
    S23_4 = (S2_3**2 + S4**2 - S2_3*S4*2*m.cos(ang[2]*rad))**0.5
    S12_5 = (S1_2**2 + S5**2 - S1_2*S5*2*m.cos(ang[3]*rad))**0.5
    S125_6 = (S12_5**2 + S6**2 - S12_5*S6*2*m.cos(ang[4]*rad))**0.5
    theta1 = m.acos((S1**2 + S1_2**2 - S2**2)/(S1*S1_2*2))/rad
    theta1p = m.acos((S2**2 + S1_2**2 - S1**2)/(S2*S1_2*2))/rad
    theta2 = m.acos((S3**2 + S2_3**2 - S2**2)/(S3*S2_3*2))/rad
    theta2p = m.acos((S2**2 + S2_3**2 - S3**2)/(S2*S2_3*2))/rad
    theta3 = m.acos((S4**2 + S23_4**2 - S2_3**2)/(S4*S23_4*2))/rad
    theta3p = m.acos((S2_3**2 + S23_4**2 - S4**2)/(S2_3*S23_4*2))/rad
    theta4p = m.acos((S1_2**2 + S12_5**2 - S5**2)/(S1_2*S12_5*2))/rad
    theta5p = m.acos((S12_5**2 + S125_6**2 - S6**2)/(S12_5*S125_6*2))/rad
    A1 = theta1 + ang[3]
    A2 = ang[0] + ang[1]
    A3 = theta2 + ang[2]
    A4 = theta3
    A5 = theta1p + theta2p + theta3p + theta4p + theta5p
    draw3__ = [A3, A4, A5]
    S3__ = [S4, S23_4, S125_6]
    
    '''第一向量(最下面邊)'''
    if A1 == 90: # 狀況一，直角
        v_x.append(S5)
        v_y.append(0)
    elif A1 == 180: # 狀況二，退化為邊 
        v_x.append(0)
        v_y.append(-1*S5)
    elif A1 >90 or A1 <90: # 狀況三，其他角 
        v_x.append(S5*m.sin(A1*rad))
        v_y.append(S5*m.cos(A1*rad))
        
    '''第二向量 S1'''
    v_x.append(0)
    v_y.append(S1)

    '''第三向量 S2 A2'''
    i_ang = 180-(ang[0] + ang[1]) # 判定角
    if i_ang==90: # 狀況一，直角
        v_x.append(S3)
        v_y.append(0)
    elif i_ang==0: # 狀況二，退化為邊 
        v_x.append(0)
        v_y.append(S3)
    elif i_ang>90 or i_ang<90: # 狀況三，其他角 
        v_x.append(S3*m.sin(A2*rad))
        v_y.append((-1)*S3*m.cos(A2*rad))
    
    '''第四向量以後'''
    if len(sid)>3:
        for j in range(len(sid)-3):
            i_ang -= draw3__[j] # 判定角
            if i_ang==0: # 負Y軸
                v_x.append(0)
                v_y.append(-1*S3__[j])
            elif i_ang==-90: # X軸
                v_x.append(S3__[j])
                v_y.append(0)
            elif i_ang==90: # 負X軸
                v_x.append(-1*S3__[j])
                v_y.append(0)
            elif i_ang==180 or i_ang==-180: # Y軸
                v_x.append(0)
                v_y.append(S3__[j])
            else: # 其他角 
                v_x.append(-1*S3__[j]*m.sin(i_ang*rad))
                v_y.append(-1*S3__[j]*m.cos(i_ang*rad))
            i_ang = i_ang + 180
    return v_x, v_y

def draw_double(sid,ang,position):
    '''計算複合形狀頂點座標'''
    rad = m.pi/180 # 轉換弧度

    '''若 position = 1，上下形狀交換'''
    temp = []
    temp2 = []
    if position == 1:
        for i in range(int(len(sid)/2)):
            temp.append(sid[i])
        for j in range(int(len(ang)/2)):
            temp2.append(ang[j])
        for k in range(int(len(sid)/2)):
            sid[k] = sid[k+int(len(sid)/2)]
        for l in range(int(len(ang)/2)):
            ang[l] = ang[l+int(len(ang)/2)]
        for n in range(int(len(sid)/2)):
            sid[n+int(len(sid)/2)] = temp[n]
        for o in range(int(len(ang)/2)):
            ang[o+int(len(ang)/2)] = temp2[o]

    vx_both = []
    vy_both = []  
    if (len(sid)/2)==2:
        '''複合三角形'''
        S1 = [sid[0],sid[2]]
        S2 = [sid[1],sid[3]]
        A1 = [ang[0],ang[1]]
        S3 = [(S1[0]**2 + S2[0]**2 - S1[0]*S2[0]*2*m.cos(A1[0]*rad))**0.5, (S1[1]**2 + S2[1]**2 - S1[1]*S2[1]*2*m.cos(A1[1]*rad))**0.5]
        A2 = [m.acos((S1[0]**2 + S3[0]**2 - S2[0]**2)/(S1[0]*S3[0]*2))/rad, m.acos((S1[1]**2 + S3[1]**2 - S2[1]**2)/(S1[1]*S3[1]*2))/rad]

        for i in range(2):
            vx = []
            vy = []
            '''尖端到垂直邊下方頂點的向量'''
            if A2[i]==90: # 狀況一，直角
                vx.append(S3[i])
                vy.append(0)
            elif A2[i]==0: # 狀況二，退化為邊 
                vx.append(0)
                vy.append(-1*S3[i])
            elif A2[i]>90 or A2[i]<90: # 狀況三，其他角 
                vx.append(S3[i]*m.sin(A2[i]*rad))
                vy.append(S3[i]*m.cos(A2[i]*rad))

            '''垂直邊向量'''
            vx.append(0)
            vy.append(S1[i])
            vx_both.append(vx)
            vy_both.append(vy)
        
    elif (len(sid)/2)==3:
        '''複合四邊形'''
        S1 = [sid[0],sid[3]]
        S2 = [sid[1],sid[4]]
        S3 = [sid[2],sid[5]]
        S4 = [(S1[0]**2 + S2[0]**2 - S1[0]*S2[0]*2*m.cos(ang[0]*rad))**0.5,(S1[1]**2 + S2[1]**2 - S1[1]*S2[1]*2*m.cos(ang[2]*rad))**0.5]
        A1 = [m.acos((S1[0]**2 + S4[0]**2 - S2[0]**2)/(S1[0]*S4[0]*2))/rad + ang[1],m.acos((S1[1]**2 + S4[1]**2 - S2[1]**2)/(S1[1]*S4[1]*2))/rad + ang[3]]
        A2 = [ang[0],ang[2]]

        '''第一向量(最下面邊)'''
        for i in range(2):
            vx = []
            vy = []
            if A1[i] == 90: # 狀況一，直角
                vx.append(S3[i])
                vy.append(0)
            elif A1[i] == 180: # 狀況二，退化為邊 
                vx.append(0)
                vy.append(-1*S3[i])
            elif A1[i] >90 or A1[i] <90: # 狀況三，其他角 
                vx.append(S3[i]*m.sin(A1[i]*rad))
                vy.append(S3[i]*m.cos(A1[i]*rad))
            
            '''第二向量 S1'''
            vx.append(0)
            vy.append(S1[i])

            '''第三向量 S2'''
            i_ang = 180 - A2[i] # 判定角
            if i_ang==90: # 狀況一，直角
                vx.append(S2[i])
                vy.append(0)
            elif i_ang==0: # 狀況二，退化為邊 
                vx.append(0)
                vy.append(S2[i])
            elif i_ang>90 or i_ang<90: # 狀況三，其他角 
                vx.append(S2[i]*m.sin(A2[i]*rad))
                vy.append((-1)*S2[i]*m.cos(A2[i]*rad))
            vx_both.append(vx)
            vy_both.append(vy)

    return vx_both[0], vy_both[0], vx_both[1], vy_both[1]

def draw_(sid_ang, changeable_side, changeable_angle, generation, num, mode, initial): #success, x__, y__ = draw_New.draw_(sid,ang, generation, i, mode, position)
    '''sid_ang: 角度與邊長參數
       changeable_side: 可變更邊數
       changeable_angle: 可變更角數
       generation: 第 N 次迭代(儲存圖片用)
       num: 第 N 次迭代中，第 X 個結構
       mode: 0 = 單一結構，1 = 複合結構
       initial: debug 用，0 = print出輪廓圖片
    '''
    sid = []
    ang = []
    for j in range(len(sid_ang)):
        if mode == 1:
            '''複合結構'''
            if j <= (changeable_side):
                sid.append(sid_ang[j])
            elif j <= (changeable_side + changeable_angle):
                ang.append(sid_ang[j])
            elif j <= (changeable_side*2 + 1 + changeable_angle):
                sid.append(sid_ang[j])
            elif j <= (changeable_side*2 + 1 + changeable_angle*2):
                ang.append(sid_ang[j])
            else:
                position = sid_ang[j]
                
        else:
            '''單一結構'''
            if j <= (changeable_side):
                sid.append(sid_ang[j])
            else:
                ang.append(sid_ang[j])
    
    print('\nsid=',sid,"    ang=",ang)
    path = os.getcwd()
    x = []
    y = []
    x2 = []
    y2 = []
    vx = []
    vy = []

    resize = 100 # 畫圖時尺度轉換
    for i in range(len(sid)):
        sid[i] = sid[i]*resize

    '''計算多邊形向量'''
    if mode == 1:
        '''複合結構'''
        vx, vy, vx2, vy2  = draw_double(sid,ang,position)

    else:
        '''單一結構'''
        if len(sid) == 2:
            '''三角形'''
            vx, vy = draw_tri(sid,ang)
            #modify
            if vx is None or vy is None:
                return 0, [], []
        
        elif len(sid) == 3:
            '''四邊形''' 
            vx, vy = draw_qua(sid,ang)
            
        elif len(sid) == 6:
            '''七邊形'''
            vx, vy = draw_hept(sid,ang)
        
    if mode == 1:
        '''複合結構的頂點位置'''
        x.append(vx[0]) # 第一點X座標
        y.append(vy[0]) # 第一點Y座標
        x2.append(vx2[0])
        y2.append(vy2[0]+resize*10)
        x.append(0) # 第二點X座標
        y.append(0) # 第二點Y座標
        x2.append(0)
        y2.append(resize*10)

        for i in range(len(vx)):
            if i > 0:
                x.append(x[i]+vx[i]) # 第N點X座標
                y.append(y[i]+vy[i]) # 第N點Y座標
                x2.append(x2[i]+vx2[i])
                y2.append(y2[i]+vy2[i])
        
        '''計算多邊形邊界，準備畫圖'''
        x_ = x     
        x_2 = x2
        if max(x2)>max(x):
            max_x = int(max(x2))
        else:
            max_x = int(max(x))
        max_y = int(max(y2)-min(y))+20*resize
        

    else:
        '''單一結構的頂點位置'''
        x.append(vx[0]) # 第一點X座標
        y.append(vy[0]) # 第一點Y座標
        x2.append(vx[0])
        y2.append(vy[0]+resize*10)
        x.append(0) # 第二點X座標
        y.append(0) # 第二點Y座標
        x2.append(0)
        y2.append(resize*10) 
        
        for i in range(len(vx)):
            if i > 0:
                x.append(x[i]+vx[i]) # 第N點X座標
                y.append(y[i]+vy[i]) # 第N點Y座標
                x2.append(x2[i]+vx[i])
                y2.append(y2[i]+vy[i])

        '''計算多邊形邊界，準備畫圖'''
        x_ = x     
        max_x = int(max(x2))
        max_y = int(max(y2)-min(y))

    '''創建邊界大小的圖片準備畫輪廓'''
    bound = 40
    depth = 30
    bound_x = max_x + 2*bound # 預留純色邊界給連通道判定
    bound_y = max_y + 2*bound # 預留純色邊界給連通道判定
    fig = np.zeros((bound_y, bound_x, 3), dtype="uint8") # 判定用變數
    fig_save = np.zeros((bound_y, bound_x, 3), dtype="uint8") # 存圖用變數
    white = (255, 255, 255)
    black = (0, 0, 0)
    
    dual = -20*resize
    thick = 3 # 線寬

    if len(vx)==2:
        '''依照座標畫三角形'''
        for j in range(len(vx)+1):
            if j < (len(vx)):
                # 判定用
                if j == 1:
                    cv2.line(fig, (bound+int(x[j]-depth), bound_y-30-int(y[j]-min(y))), (bound+int(x2[j+1]-depth), bound_y-30-int(y2[j+1]-min(y))), white, thick) # 左豎
                    cv2.line(fig, (bound+int(x2[j]), bound_y-30-int(y2[j]-min(y))), (bound+int(x[j+1]), bound_y-30-int(y[j+1]-min(y))), white, thick) # 右豎
                    cv2.line(fig, (bound+int(x[j]), bound_y-30-int(y[j]-min(y))), (bound+int(x[j]-depth), bound_y-30-int(y[j]-min(y))), white, thick) # 上橫
                    cv2.line(fig, (bound+int(x2[j+1]), bound_y-30-int(y2[j+1]-min(y))), (bound+int(x2[j+1]-depth), bound_y-30-int(y2[j+1]-min(y))), white, thick) # 下橫
                    cv2.line(fig_save, (bound+int(x2[j]), bound_y-30-int(y2[j]-min(y))), (bound+int(x[j+1]), bound_y-30-int(y[j+1]-min(y))), white, thick) # 右豎
                else:
                    cv2.line(fig, (bound+int(x[j]), bound_y-30-int(y[j]-min(y))), (bound+int(x[j+1]), bound_y-30-int(y[j+1]-min(y))), white, thick)
                    cv2.line(fig, (bound+int(x2[j]), bound_y-30-int(y2[j]-min(y))), (bound+int(x2[j+1]), bound_y-30-int(y2[j+1]-min(y))), white, thick)
                # 存圖用
                cv2.line(fig_save, (bound+int(x[j]), bound_y-30-int(y[j]-min(y))), (bound+int(x[j+1]), bound_y-30-int(y[j+1]-min(y))), white, thick)
                cv2.line(fig_save, (bound+int(x2[j]), bound_y-30-int(y2[j]-min(y))), (bound+int(x2[j+1]), bound_y-30-int(y2[j+1]-min(y))), white,thick)
            else:
                # 判定用
                cv2.line(fig, (bound+int(x[j]), bound_y-30-int(y[j]-min(y))), (bound+int(x[0]), bound_y-30-int(y[0]-min(y))), white, thick)
                cv2.line(fig, (bound+int(x2[j]), bound_y-30-int(y2[j]-min(y))), (bound+int(x2[0]), bound_y-30-int(y2[0]-min(y))), white, thick)
                # 存圖用
                cv2.line(fig_save, (bound+int(x[j]), bound_y-30-int(y[j]-min(y))), (bound+int(x[0]), bound_y-30-int(y[0]-min(y))), white, thick)
                cv2.line(fig_save, (bound+int(x2[j]), bound_y-30-int(y2[j]-min(y))), (bound+int(x2[0]), bound_y-30-int(y2[0]-min(y))), white, thick)

        if mode == 1:
            '''依照座標畫複合三角形'''
            for j in range(len(vx)+1):
                if j < (len(vx)):
                    # 判定用
                    if j == 1:
                        cv2.line(fig, (bound+int(x2[2]), bound_y-30-int(y2[2]-min(y))), (bound+int(x2[2]-depth), bound_y-30-int(y2[2]-min(y))), black, thick) # 下橫去掉
                        cv2.line(fig, (bound+int(x2[2]), bound_y-30-int(y2[2]-min(y))), (bound+int(x[1]), dual+bound_y-30-int(y[1]-min(y))), white, thick) # 右豎連接
                        cv2.line(fig, (bound+int(x[1]-depth),  bound_y-20-int(y2[1]-min(y))), (bound+int(x2[1]-depth), dual+bound_y-30-int(y2[2]-min(y))), white, thick) # 左豎
                        cv2.line(fig, (bound+int(x2[2]), dual+bound_y-30-int(y[2]-min(y))), (bound+int(x[2]), dual+bound_y-30-int(y2[1]-min(y))), white, thick) # 右豎
                        cv2.line(fig, (bound+int(x2[2]), dual+bound_y-30-int(y2[2]-min(y))), (bound+int(x2[2]-depth), dual+bound_y-30-int(y2[2]-min(y))), white, thick) # 下橫
                        cv2.line(fig_save, (bound+int(x2[1]), dual+bound_y-30-int(y[2]-min(y))), (bound+int(x2[2]), dual+bound_y-30-int(y2[2]-min(y))), white, thick) # 右豎
                        
                    else:
                        cv2.line(fig, (bound+int(x[j]), dual+bound_y-30-int(y[j]-min(y))), (bound+int(x[j+1]), dual+bound_y-30-int(y[j+1]-min(y))), white, thick)
                        cv2.line(fig, (bound+int(x2[j]), dual+bound_y-30-int(y2[j]-min(y))), (bound+int(x2[j+1]), dual+bound_y-30-int(y2[j+1]-min(y))), white, thick)
                    # 存圖用
                    cv2.line(fig_save, (bound+int(x[j]), dual+bound_y-30-int(y[j]-min(y))), (bound+int(x[j+1]), dual+bound_y-30-int(y[j+1]-min(y))), white, thick)
                    cv2.line(fig_save, (bound+int(x2[j]), dual+bound_y-30-int(y2[j]-min(y))), (bound+int(x2[j+1]), dual+bound_y-30-int(y2[j+1]-min(y))), white, thick)
                    
                else:
                    # 判定用
                    cv2.line(fig, (bound+int(x[j]), dual+bound_y-30-int(y[j]-min(y))), (bound+int(x[0]), dual+bound_y-30-int(y[0]-min(y))), white, thick)
                    cv2.line(fig, (bound+int(x2[j]), dual+bound_y-30-int(y2[j]-min(y))), (bound+int(x2[0]), dual+bound_y-30-int(y2[0]-min(y))), white, thick)
                    # 存圖用
                    cv2.line(fig_save, (bound+int(x[j]), dual+bound_y-30-int(y[j]-min(y))), (bound+int(x[0]), dual+bound_y-30-int(y[0]-min(y))), white, thick)
                    cv2.line(fig_save, (bound+int(x2[j]), dual+bound_y-30-int(y2[j]-min(y))), (bound+int(x2[0]), dual+bound_y-30-int(y2[0]-min(y))), white, thick)

    else:
        '''依照座標畫多邊形'''
        for j in range(len(vx)+1):
            if j < (len(vx)):
                # 判定用
                if j == 1:
                    cv2.line(fig, (bound+int(x[j]-depth), bound_y-30-int(y[j]-min(y))), (bound+int(x2[j+1]-depth), bound_y-30-int(y2[j+1]-min(y))), white, thick) # 左豎
                    cv2.line(fig, (bound+int(x2[j]), bound_y-30-int(y2[j]-min(y))), (bound+int(x[j+1]), bound_y-30-int(y[j+1]-min(y))), white, thick) # 右豎
                    cv2.line(fig, (bound+int(x[j]), bound_y-30-int(y[j]-min(y))), (bound+int(x[j]-depth), bound_y-30-int(y[j]-min(y))), white, thick) # 上橫
                    cv2.line(fig, (bound+int(x2[j+1]), bound_y-30-int(y2[j+1]-min(y))), (bound+int(x2[j+1]-depth), bound_y-30-int(y2[j+1]-min(y))), white, thick) # 下橫
                    cv2.line(fig_save, (bound+int(x2[j]), bound_y-30-int(y2[j]-min(y))), (bound+int(x[j+1]), bound_y-30-int(y[j+1]-min(y))), white, thick) # 右豎
                else:
                    cv2.line(fig, (bound+int(x[j]), bound_y-30-int(y[j]-min(y))), (bound+int(x[j+1]), bound_y-30-int(y[j+1]-min(y))), white, thick)
                    cv2.line(fig, (bound+int(x2[j]), bound_y-30-int(y2[j]-min(y))), (bound+int(x2[j+1]), bound_y-30-int(y2[j+1]-min(y))), white, thick)
                # 存圖用
                cv2.line(fig_save, (bound+int(x[j]), bound_y-30-int(y[j]-min(y))), (bound+int(x[j+1]), bound_y-30-int(y[j+1]-min(y))), white, thick)
                cv2.line(fig_save, (bound+int(x2[j]), bound_y-30-int(y2[j]-min(y))), (bound+int(x2[j+1]), bound_y-30-int(y2[j+1]-min(y))), white, thick)
                
            else:
                # 判定用
                cv2.line(fig, (bound+int(x[j]), bound_y-30-int(y[j]-min(y))), (bound+int(x[0]), bound_y-30-int(y[0]-min(y))), white, thick)
                cv2.line(fig, (bound+int(x2[j]), bound_y-30-int(y2[j]-min(y))), (bound+int(x2[0]), bound_y-30-int(y2[0]-min(y))), white, thick)
                # 存圖用
                cv2.line(fig_save, (bound+int(x[j]), bound_y-30-int(y[j]-min(y))), (bound+int(x[0]), bound_y-30-int(y[0]-min(y))), white, thick)
                cv2.line(fig_save, (bound+int(x2[j]), bound_y-30-int(y2[j]-min(y))), (bound+int(x2[0]), bound_y-30-int(y2[0]-min(y))), white, thick)
        
        if mode == 1:
            '''依照座標畫複合多邊形'''
            for j in range(len(vx)+1):
                if j < (len(vx)):
                    # 判定用
                    if j == 1:
                        cv2.line(fig, (bound+int(x2[2]), bound_y-30-int(y2[2]-min(y))), (bound+int(x2[2]-depth), bound_y-30-int(y2[2]-min(y))), black, thick) # 結構1下橫去掉
                        cv2.line(fig, (bound+int(x2[2]), bound_y-30-int(y2[2]-min(y))), (bound+int(x[1]), dual+bound_y-30-int(y[1]-min(y))), white, thick) # 右豎連接
                        cv2.line(fig, (bound+int(x[1]-depth),  bound_y-20-int(y2[1]-min(y))), (bound+int(x2[1]-depth), dual+bound_y-30-int(y2[2]-min(y))), white, thick) # 左豎
                        cv2.line(fig, (bound+int(x2[2]), dual+bound_y-30-int(y[2]-min(y))), (bound+int(x[2]), dual+bound_y-30-int(y2[1]-min(y))), white, thick) # 右豎
                        cv2.line(fig, (bound+int(x2[2]), dual+bound_y-30-int(y2[2]-min(y))), (bound+int(x2[2]-depth), dual+bound_y-30-int(y2[2]-min(y))), white, thick) # 結構2下橫
                        cv2.line(fig, (bound+int(x[0]), dual+bound_y-30-int(y[0]-min(y))), (bound+int(x[3]), dual+bound_y-30-int(y[3]-min(y))), white, thick)
                        cv2.line(fig, (bound+int(x2[0]), dual+bound_y-30-int(y2[0]-min(y))), (bound+int(x2[3]), dual+bound_y-30-int(y2[3]-min(y))), white, thick)
                        cv2.line(fig_save, (bound+int(x2[j]), dual+bound_y-30-int(y2[j]-min(y))), (bound+int(x[j+1]), dual+bound_y-30-int(y[j+1]-min(y))), white, thick) # 右豎
                    else:
                        cv2.line(fig, (bound+int(x[j]), dual+bound_y-30-int(y[j]-min(y))), (bound+int(x[j+1]), dual+bound_y-30-int(y[j+1]-min(y))), white, thick)
                        cv2.line(fig, (bound+int(x2[j]), dual+bound_y-30-int(y2[j]-min(y))), (bound+int(x2[j+1]), dual+bound_y-30-int(y2[j+1]-min(y))), white, thick)
                    
                    # 存圖用
                    cv2.line(fig_save, (bound+int(x[j]), dual+bound_y-30-int(y[j]-min(y))), (bound+int(x[j+1]), dual+bound_y-30-int(y[j+1]-min(y))), white, thick)
                    cv2.line(fig_save, (bound+int(x2[j]), dual+bound_y-30-int(y2[j]-min(y))), (bound+int(x2[j+1]), dual+bound_y-30-int(y2[j+1]-min(y))), white, thick)

                else:
                    # 判定用
                    cv2.line(fig, (bound+int(x[j]), dual+bound_y-30-int(y[j]-min(y))), (bound+int(x[0]), dual+bound_y-30-int(y[0]-min(y))), white, thick)
                    cv2.line(fig, (bound+int(x2[j]), dual+bound_y-30-int(y2[j]-min(y))), (bound+int(x2[0]), dual+bound_y-30-int(y2[0]-min(y))), white, thick)
                    # 存圖用
                    cv2.line(fig_save, (bound+int(x[j]), dual+bound_y-30-int(y[j]-min(y))), (bound+int(x[0]), dual+bound_y-30-int(y[0]-min(y))), white, thick)
                    cv2.line(fig_save, (bound+int(x2[j]), dual+bound_y-30-int(y2[j]-min(y))), (bound+int(x2[0]), dual+bound_y-30-int(y2[0]-min(y))), white, thick)

    cv2.imwrite('plot.png', fig)


    second_tri = -1
    '''畫多邊形的分割線'''
    if len(vx)==3:
        '''四邊形'''
        if mode == 1:
            '''複合四邊形'''
            cv2.line(fig_save, (bound+int(x[3]), bound_y-30-int(y[3]-min(y))), (bound+int(x[1]), bound_y-30-int(y[1]-min(y))), white, thick)
            cv2.line(fig_save, (bound+int(x2[3]), bound_y-30-int(y2[3]-min(y))), (bound+int(x2[1]), bound_y-30-int(y2[1]-min(y))), white, thick)
            cv2.line(fig_save, (bound+int(x[3]), dual+bound_y-30-int(y[3]-min(y))), (bound+int(x[1]), dual+bound_y-30-int(y[1]-min(y))), white, thick)
            cv2.line(fig_save, (bound+int(x2[3]), dual+bound_y-30-int(y2[3]-min(y))), (bound+int(x2[1]), dual+bound_y-30-int(y2[1]-min(y))), white, thick)
        else:
            '''四邊形'''
            if second_tri == 1:# 第二個三角形在上
                cv2.line(fig_save, (bound+int(x[2]), bound_y-30-int(y[2]-min(y))), (bound+int(x[0]), bound_y-30-int(y[0]-min(y))), white, thick)
                cv2.line(fig_save, (bound+int(x2[2]), bound_y-30-int(y2[2]-min(y))), (bound+int(x2[0]), bound_y-30-int(y2[0]-min(y))), white, thick)
            else:
                cv2.line(fig_save, (bound+int(x[3]), bound_y-30-int(y[3]-min(y))), (bound+int(x[1]), bound_y-30-int(y[1]-min(y))), white, thick)
                cv2.line(fig_save, (bound+int(x2[3]), bound_y-30-int(y2[3]-min(y))), (bound+int(x2[1]), bound_y-30-int(y2[1]-min(y))), white, thick)
       
    elif len(vx) == 6:
        '''七邊形'''
        cv2.line(fig_save, (bound+int(x[0]), bound_y-30-int(y[0]-min(y))), (bound+int(x[5]), bound_y-30-int(y[5]-min(y))), white, thick)
        cv2.line(fig_save, (bound+int(x[1]), bound_y-30-int(y[1]-min(y))), (bound+int(x[5]), bound_y-30-int(y[5]-min(y))), white, thick)
        cv2.line(fig_save, (bound+int(x[2]), bound_y-30-int(y[2]-min(y))), (bound+int(x[5]), bound_y-30-int(y[5]-min(y))), white,thick)
        cv2.line(fig_save, (bound+int(x[3]), bound_y-30-int(y[3]-min(y))), (bound+int(x[5]), bound_y-30-int(y[5]-min(y))), white, thick)
        cv2.line(fig_save, (bound+int(x2[0]), bound_y-30-int(y2[0]-min(y))), (bound+int(x2[5]), bound_y-30-int(y2[5]-min(y))), white, thick)
        cv2.line(fig_save, (bound+int(x2[1]), bound_y-30-int(y2[1]-min(y))), (bound+int(x2[5]), bound_y-30-int(y2[5]-min(y))), white, thick)
        cv2.line(fig_save, (bound+int(x2[2]), bound_y-30-int(y2[2]-min(y))), (bound+int(x2[5]), bound_y-30-int(y2[5]-min(y))), white, thick)
        cv2.line(fig_save, (bound+int(x2[3]), bound_y-30-int(y2[3]-min(y))), (bound+int(x2[5]), bound_y-30-int(y2[5]-min(y))), white, thick)

    cv2.imwrite('plot_.png', fig_save)

    '''debug 用，0 = print出輪廓圖片'''
    if initial == 0:
        cv2.imshow("fig", fig_save)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

    '''計算連通道、存圖檔'''
    img = cv2.imread(path + "\\plot.png", cv2.IMREAD_GRAYSCALE)
    ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    ret,thresh_INV = cv2.threshold(thresh,127,255,cv2.THRESH_BINARY_INV)
    cv2.imwrite(path + "\\img.png", thresh_INV) # 當次輪廓的圖，會一直被覆蓋掉
    cv2.imwrite(path + '\\bound\\img_'+str(generation+1)+'-'+str(num+1)+'.png', thresh_INV) # 紀錄所有輪廓的圖，存檔觀察用

    img2 = cv2.imread(path + "\\plot_.png", cv2.IMREAD_GRAYSCALE)
    ret2,thresh2 = cv2.threshold(img2,127,255,cv2.THRESH_BINARY)
    ret2,thresh_INV2 = cv2.threshold(thresh2,127,255,cv2.THRESH_BINARY_INV)
    cv2.imwrite(path + '\\plot\\img_'+str(generation+1)+'-'+str(num+1)+'.png', thresh_INV2) # 紀錄所有含有分割線之輪廓的圖，存檔觀察用
    num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(thresh_INV, connectivity=8, ltype=cv2.CV_32S) # 白底
    num_labels_, labels, stats, centers = cv2.connectedComponentsWithStats(thresh, connectivity=8, ltype=cv2.CV_32S) # 黑底 暫無用
    
    '''建模可行性判定--頂點有無過界'''
    success = 1
    if mode == 1:
        if min(x_)<0 or min(x_2)<0:
            '''過界(X座標<0)'''
            success = 0
            print("failed 無效幾何, 過界(X<0)")
    else:
        if min(x_)<0:
            '''過界(X座標<0)'''
            success = 0
            print("failed 無效幾何, 過界(X<0)")
                
    # elif num_labels_<2:
    #     success = 0 # 黑底連通道數量要=2
    #     print("failed 結構線交叉重疊, num_labels = 3 but",num_labels_)
   
    '''建模可行性判定--輪廓有無重疊'''
    if success == 1 and num_labels>3:
        '''白底連通道數量要 = 3'''
        success = 0
        print("failed 結構線重疊, num_labels_INV = 3 but",num_labels)
    elif success == 1:
        print("success")
   
    '''其中一項沒通過就是無法建模'''
    if mode == 1:
        x = [x, x2]
        y = [y, y2]
    
    return success, x, y

def plotting(lines_x, lines_y):
    '''迭代完成後，劃出一百次中各次的最高分'''
    path = os.getcwd()
    plt.cla()
    fig1 = plt.figure(1)
    plt.ylabel('score')
    plt.xlabel('generation')
    plt.plot(lines_x, lines_y) # 各點座標
    plt.xlim((0, 100)) # X軸範圍-世代
    # plt.ylim((np.min(lines_y), np.max(lines_y))) # Y軸範圍-最佳值
    plt.ylim((0, np.max(lines_y)+5)) # Y軸範圍-最佳值
    plt.ioff()
    plt.savefig(path + "\\output-N邊形迭代結果折線圖.png")
    plt.draw()
    plt.pause(3)
    plt.close(fig1)