# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 13:58:46 2020

@author: ck101

本檔案函式功能為控制光學模擬軟體TracePro，將模擬需要的視窗打開，執行定義好的 Macro 檔案，
輸出可視化圖像、極座標坎德拉圖、長方坎德拉圖與 txt 檔。
操控軟體介面所使用的指令可以搜尋 pywinauto 套件。
"""
import pywinauto
import time
import warnings
import pyscreenshot as ImageGrab
import cv2
import os

def tracepro(population_number):
    warnings.simplefilter('ignore', category=UserWarning)
    
    '''連接 TracePro 應用程式'''
    app = pywinauto.application.Application().connect(path="C:\\Program Files (x86)\\Lambda Research Corporation\\TracePro\\TracePro.exe") # 連接
    #app = pywinauto.application.Application().Start(cmd_line=u'"C:\\Program Files (x86)\\Lambda Research Corporation\\TracePro\\TracePro.exe" ') # 啟動
    afxa = app.TracePro # 連接 TracePro
    try:
        afxa.wait('ready')
    except:
        print("timed out") # pywinauto.timings.TimeoutError: timed out
    
    '''開啟 Polar Candela Distribution 及 Rectangular Candela Distribution'''
    menu_item4 = afxa.menu_item(u'&Analysis->&Candela Plots->Polar &Candela Distribution').select()
    afxframeorview = afxa.AfxFrameOrView42
    afxframeorview.maximize() # 放大視窗
    time.sleep(0.2)
    afxb = app[u'Afx:400000:8:10003:0:b3c0979'].wait('ready')
    afxb.menu_item(u'&Analysis->&Candela Plots->Rectangular Candela &Distribution').select()
    time.sleep(1)

    '''選取主要啟動 Macro 檔的 OML 視窗'''
    afxc = app[u'Afx:400000:8:10003:0:bf0aa1'].wait('ready')
    afxc.menu_item(u'&Window->&1 Model:[initiate.OML]').select()
    
    '''啟動 Macro 檔'''
    afxe = app[u'Afx:400000:8:10003:0:cc0667']
    afxe.menu_item(u'&Macros->&Execute').select()
    #w_handle1 = pywinauto.findwindows.find_windows(title=u'Select macro file to Load/Execute',class_name="#32770")[0]
    #print("tracepro w_handle1",w_handle1)
    #window1 = app.window(handle=w_handle1)
    '''modify by po-yu'''
    for find_Select_macro_file_window in range(3):
        try:
            w_handle1 = pywinauto.findwindows.find_windows(title=u'Select macro file to Load/Execute',class_name="#32770")[0]
            window1 = app.window(handle=w_handle1)
            break
        except:                  
            print('尋找 Select macro file to Load/Execute 視窗第'+str(find_Select_macro_file_window+1)+'次發生錯誤，將重試至第3次')
            time.sleep(1)
    '''modify by po-yu'''
    
    '''打開指定資料夾內的 Macro 檔'''
    j = population_number+1
    path = os.getcwd()
    window1[u'檔案名稱(&N):Edit'].type_keys(r""+ path + "\GA_population\P"+str(j)+r"\Fin.scm")
    #window1[u'檔案名稱(&N):Edit'].type_keys(r""+ path + "\GA_population\P"+str(j)+r"\Final_run.scm")
    time.sleep(2)
    window1[u'開啟(&O)Button'].click()
    
    '''確認是否已經按下執行鍵運行 Macro 檔'''
    check_run = 1
    while check_run>0:
        try:
            w_handle1 = pywinauto.findwindows.find_windows(title=u'Select macro file to Load/Execute',class_name="#32770")[0]#modify w_handle2
            window1 = app.window(handle=w_handle1)
            window1[u'開啟(&O)Button'].click()
        except IndexError:
            print("已開啟Macrow執行模擬")
            check_run = -1
    
    wait_time = 6 # 模擬啟動後等待 6 秒後進行確認模擬完成與否
    time.sleep(wait_time)


    OK_or_NOT = 0 # 是否完成整個光線追蹤模擬流程(0=失敗 1=成功)，有可能參數組雖然通過建模可行性判定，由於結構間邊線過於接近，導致建模時無法拉伸實體模型，但還是會執行模擬。
    check_end = 1 # 確認是否完成流程，視窗回到初始設定(只有一個非全屏視窗)
    check_num = 1 # 紀錄確認次數
    '''確認是否完成流程，此步是以擷取畫面偏右上方的圖塊，確認軟體視窗以回到初始化狀態(灰色圖塊)便判定為成功，
       因模擬失敗，macro 流程會卡住，導致視窗維持全螢幕狀態，此時擷取的圖塊呈白色。'''
    while check_end>0:
        '''擷取指定範圍畫面，影像大小為 1920x1080 像素(需依照螢幕解析度調整數字)'''
        im = ImageGrab.grab(bbox=(0,#90,  # X1
            950,#200,   # Y1
            10,#100,  # X2
            960,#210 # Y2
            ))  

        '''儲存檔案並以灰階方式讀取'''
        im.save("box.png")
        time.sleep(1)
        img = cv2.imread(path + "\\box.png", cv2.IMREAD_GRAYSCALE)
        print("waiting")

        '''計算範圍內一百個像素平均灰階值'''
        pixel = 0
        for i in range(10):
            for j in range(10):
                pixel += img[i][j]
        pixel = pixel/100
        print("pixel",pixel)
        '''若灰階職小於兩百，代表視窗已回到初始設定(擷取區域會是深灰色)，如果為初始化，會維持 255 的白色'''
        if pixel<200 and pixel>160:
            print("Tracepro pix<200 OK")
            check_end = -1 # 確認該次流程成功，結束
            OK_or_NOT = 1 # 模擬分析成功，數值轉為 1
            time.sleep(2) # 隔兩秒確認一次
        check_num += 1 # 累計確認次數
        
        '''若確認次數大於 10 次，執行用於初始化模擬環境的 Macro 檔'''
        if check_num > 13:
            '''啟動初始化模擬環境的 Macro 檔'''
            afxe.menu_item(u'&Macros->&Execute').select()
            w_handle1 = pywinauto.findwindows.find_windows(title=u'Select macro file to Load/Execute',class_name="#32770")[0]
            window1 = app.window(handle=w_handle1)
            window1[u'檔案名稱(&N):Edit'].type_keys(r""+path+"\Data\Reset.scm")
            time.sleep(2)
            window1[u'開啟(&O)Button'].click()
            
            '''確認是否已經按下執行鍵運行 Macro 檔'''
            check_run = 1
            while check_run>0:
                try:
                    w_handle1 = pywinauto.findwindows.find_windows(title=u'Select macro file to Load/Execute',class_name="#32770")[0]
                    window1 = app.window(handle=w_handle1)#modify w_handle2
                    window1[u'開啟(&O)Button'].click()
                except IndexError:
                    print("已開啟Macrow執行模擬")
                    check_run = -1
            check_end = -1 # 確認該次流程失敗，結束
    return OK_or_NOT

#tracepro(0, 1)