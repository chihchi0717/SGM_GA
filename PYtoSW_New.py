# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 13:58:46 2020

@author: ck101

本檔案函式功能為控制3D繪圖軟體SolidWorks，新建零件檔案，將通過建模可行性判定的參數組建出稜鏡模型，存在指定位置。
"""

import time
import pythoncom
import win32com.client
import os

# Solidworks Version (2016)
swYearLastDigit = 6

async def Build_model(x_, y_, success_num, mode):
    '''x_: 得到的X座標
       y_: 得到的Y座標
       success_num: 成功建模次數(關閉視窗用)
       mode: 0 = 單一結構，1 = 複合結構
    '''
    N = 130 # 結構陣列數
    x = []
    y = []

    if mode == 1:
        '''複合結構，分別記錄兩個結構的座標'''
        for i in range(2):
            x0 = []
            y0 = []
            for j in range(len(x_[i])):
                x0.append(round((x_[i][j]/10000), 5))
                y0.append(round((y_[i][j]/10000), 5))
            x.append(x0)
            y.append(y0)
    else:
        '''單一結構，只需記錄一個結構的座標'''
        for i in range(len(x_)):
            x.append(round((x_[i]/10000), 5))
            y.append(round((y_[i]/10000), 5))
            
    '''對照軟體版本'''
    sw = win32com.client.Dispatch("SldWorks.Application.{}".format((20+(swYearLastDigit-2)))) # e.g. 20 is SW2012,  27 is SW2019

    '''Creating new document - part'''
    folder = 'C:\\ProgramData\\SolidWorks\\SOLIDWORKS 2016\\templates\\'
    name = '零件.prtdot'
    filename = os.path.join(folder, name) # 匯入起始空白零件檔
    Model = sw.NewDocument(filename, 0, 0, 0) # params: template, ?, sheet_width,sheet_height
    
    Model = sw.ActiveDoc
    ARG_NULL = win32com.client.VARIANT(pythoncom.VT_DISPATCH, None) # Initializing application
    '''SelectByID2(‘Name’, ‘Type’, ‘X’, ‘Y’, ‘Z’, ‘Append’, ‘Mark’, ‘Callout’, ‘SelectOption)'''
    Model.Extension.SelectByID2("前基準面", "PLANE", 0, 0, 0, False, 0, ARG_NULL, 0) # 選擇前基準面
    Model.SketchManager.InsertSketch(True) # 進行草圖
    Model.ClearSelection2(True) # 清空指令
    
    '''---畫結構輪廓---'''
    if mode == 1:
        '''畫複合結構'''
        for i in range(len(x[0])):
            if i==len(x[0])-1: # 最後一筆連接成封閉面
                '''CreateLine(‘Xstart’, ‘Ystart’, ‘Zstart’, ‘Xstop’, ‘Ystop’, ‘Zstop’) 單位M'''
                Model.SketchManager.CreateLine(x[0][i], y[0][i], 0, x[0][0], y[0][0], 0)
                '''CreateLinearSketchStepAndRepeat(NumX,NumY,SpacingX,SpacingY,AngleX,AngleY,DeleteInstances,
                    XSpacingDim,YSpacingDim, AngleDim, CreateNumOfInstancesDimInXDir, CreateNumOfInstancesDimInYDir)'''
                Model.SketchManager.CreateLinearSketchStepAndRepeat(1, N/2, 0.1, 0.2, 0, 1.5707963267949, "", False, False, False, False, True)
                Model.SketchManager.CreateLine(x[1][i], y[1][i], 0, x[1][0], y[1][0], 0)
                Model.SketchManager.CreateLinearSketchStepAndRepeat(1, N/2, 0.1, 0.2, 0, 1.5707963267949, "", False, False, False, False, True)
            elif i == 1: # 畫垂直面
                if y[0][2] < 0.1:
                    Model.SketchManager.CreateLine(0, y[0][2], 0, 0, 0.1, 0)
                    Model.SketchManager.CreateLinearSketchStepAndRepeat(1, N/2, 0.1, 0.2, 0, 1.5707963267949, "", False, False, False, False, True) # N-1
                if y[1][2] < 0.2:   
                    Model.SketchManager.CreateLine(0, y[1][2], 0, 0, 0.2, 0)
                    Model.SketchManager.CreateLinearSketchStepAndRepeat(1, N/2, 0.1, 0.2, 0, 1.5707963267949, "", False, False, False, False, True) # N-1
            else: # 畫與垂直面以外斜面
                j = i+1
                Model.SketchManager.CreateLine(x[0][i], y[0][i], 0, x[0][j], y[0][j], 0)
                #print(x[i], y[i],x[j], y[j])
                Model.SketchManager.CreateLinearSketchStepAndRepeat(1, N/2, 0.1, 0.2, 0, 1.5707963267949, "", False, False, False, False, True)
                Model.SketchManager.CreateLine(x[1][i], y[1][i], 0, x[1][j], y[1][j], 0)
                Model.SketchManager.CreateLinearSketchStepAndRepeat(1, N/2, 0.1, 0.2, 0, 1.5707963267949, "", False, False, False, False, True)
            
    else:
        '''畫單一結構'''
        for i in range(len(x)):
            if i==len(x)-1:
                Model.SketchManager.CreateLine(x[i], y[i], 0, x[0], y[0], 0)
                Model.SketchManager.CreateLinearSketchStepAndRepeat(1, N, 0.1, 0.1, 0, 1.5707963267949, "", False, False, False, False, True)
            elif i == 1:
                if y[2] < 0.1:
                    Model.SketchManager.CreateLine(0, y[2], 0, 0, 0.1, 0)
                    Model.SketchManager.CreateLinearSketchStepAndRepeat(1, N, 0.1, 0.1, 0, 1.5707963267949, "", False, False, False, False, True) # N-1
            else:
                j = i+1
                Model.SketchManager.CreateLine(x[i], y[i], 0, x[j], y[j], 0)
                Model.SketchManager.CreateLinearSketchStepAndRepeat(1, N, 0.1, 0.1, 0, 1.5707963267949, "", False, False, False, False, True)
   
    Model.ClearSelection2(True) # 清空滑鼠功能
    '''---畫結構輪廓END---'''
    
    
    '''畫殘留層'''
    H = N*0.1 #(N-1)*0.1+y[2]
    Model.SketchManager.CreateLine(0, 0, 0, -0.01, 0, 0) # 下
    Model.SketchManager.CreateLine(-0.01, 0, 0, -0.01, H, 0) # 直
    Model.SketchManager.CreateLine(-0.01, H, 0, 0, H, 0) # 上
    Model.ClearSelection2(True)

    '''畫檔板(可不用)'''
    Model.SketchManager.CreateLine(0, H+0.005, 0, -0.01, H+0.005, 0) # 下
    Model.SketchManager.CreateLine(-0.01, H+0.005, 0, -0.01, H+0.31, 0) # 左直
    Model.SketchManager.CreateLine( -0.01, H+0.31, 0,  0, H+0.31, 0) # 上
    Model.SketchManager.CreateLine(0, H+0.31, 0, 0, H+0.005, 0) # 右直
    Model.ClearSelection2(True)

    '''畫光源'''
    '''光源尺寸'''
    H2 = 10
    H2_ = 0.4
    W2 = 0.7
    W2_ = 0.1
    Model.SketchManager.CreateLine(W2, H2, 0, W2+W2_, H2, 0)
    Model.SketchManager.CreateLine(W2+W2_, H2, 0, W2+W2_, H2+H2_, 0)
    Model.SketchManager.CreateLine(W2+W2_, H2+H2_, 0, W2, H2+H2_, 0)
    Model.SketchManager.CreateLine(W2, H2+H2_, 0, W2, H2, 0)
    Model.ClearSelection2(True)

    '''實體拉伸'''
    Model.FeatureManager.FeatureExtrusion2(True, False, False,0,0,1, # 厚度M
        0,False,False,False,False,0,0,False,False,False,False,True,True,True,0,0,False,)
    Model.SelectionManager.EnableContourSelection = False
    Model.ClearSelection2(True)

    
    print("Model has been builded, wait 0.3 sec to save SAT File")
    time.sleep(0.3)
    path = os.getcwd()
    Model.SaveAs3(path + "\\file\\prism_sat_file.SAT", 0, 0)
    print(success_num) # 成功建模次數
    if success_num == 10: # 成功建模累計前十次後，將這十個視窗關閉 
        for k in range(10):
            certen_name = ('零件'+str(k+1))
            print("SW",certen_name)
            Model = sw.CloseDoc(certen_name)

    elif success_num%10 == 0: # 每成功建模累計十次，將這十個視窗關閉
        for m in range(9):
            certen_name2 = ('零件'+str(int(success_num/10)-1)+str(int(m+1)))
            print("SW",certen_name2)
            Model = sw.CloseDoc(certen_name2)
        Model = sw.CloseDoc(('零件'+str(int(success_num/10))+str(0)))
    return 1

# try:
#     if __name__ == "__main__":
#         # asyncio.run(Build_model([0,1],1,1))
#         # x=[5, 0, 0, 3, 3]
#         # y=[5, 0, 7, 7, 0]
#         # sid= [1.7918397477265013, 0, 0, 2.1196770569328196, 1.1453069921475842] 
#         # ang= [4.667902132486009, 0, 8, 4.607807615374297, 4.382856561030431]   
#         # x= [0.5, 0, 0, 4.29, 3.73]
#         # y= [0.87, 0, 10, 1.42, -0.5] 
#         x_ = [[482.9629131445342, 0, 0], [565.685424949238, 0, 0]]
#         y_ = [[129.40952255126044, 0, 900], [1565.685424949238, 1000, 1100]]
#         asyncio.get_event_loop().run_until_complete(Build_model(x_, y_,1,1,1))
        
# except KeyboardInterrupt:
#     sys.exit()
