import pywinauto
import time
import warnings
import os


def tracepro(population_number):
    sleep_time = 0.5
    warnings.simplefilter('ignore', category=UserWarning)
    
    '''檢查是否存在 start.OML 或 completion_signal.OML，進行相應操作'''
    signal_file_path = "C:/Users/cchih/Desktop/NTHU/MasterThesis/GA/file/completion_signal.OML"

    if os.path.exists(signal_file_path):
        os.remove(signal_file_path)
        print(f"{signal_file_path} 已刪除")
    else:
        print(f"{signal_file_path} 不存在，無需刪除")
    

    '''連接 TracePro 應用程式'''
    app = pywinauto.application.Application().connect(path="C:\\Program Files (x86)\\Lambda Research Corporation\\TracePro\\TracePro.exe") # 連接
    afxa = app.TracePro # 連接 TracePro
    try:
        afxa.wait('ready')
    except:
        print("timed out") # pywinauto.timings.TimeoutError: timed out
    time.sleep(sleep_time)
    
    '''開啟 Polar Candela Distribution 及 Rectangular Candela Distribution'''
    menu_item4 = afxa.menu_item(u'&Analysis->&Candela Plots->Polar &Candela Distribution').select()
    afxframeorview = afxa.AfxFrameOrView42
    afxframeorview.maximize() # 放大視窗
    time.sleep(sleep_time)
    afxb = app[u'Afx:400000:8:10003:0:b3c0979'].wait('ready')
    afxb.menu_item(u'&Analysis->&Candela Plots->Rectangular Candela &Distribution').select()
    time.sleep(sleep_time)

    '''選取主要啟動 Macro 檔的 OML 視窗'''
    afxc = app[u'Afx:400000:8:10003:0:bf0aa1'].wait('ready')
    afxc.menu_item(u'&Window->&1 Model:[initiate.OML]').select()
    
    '''啟動 Macro 檔'''
    afxe = app[u'Afx:400000:8:10003:0:cc0667']
    afxe.menu_item(u'&Macros->&Execute').select()
    
    '''打開指定資料夾內的 Macro 檔'''
    j = population_number + 1
    path = os.getcwd()
    w_handle1 = pywinauto.findwindows.find_windows(title=u'Select macro file to Load/Execute', class_name="#32770")[0]
    window1 = app.window(handle=w_handle1)
    window1[u'檔案名稱(&N):Edit'].type_keys(r"" + path + "\GA_population\P" + str(j) + r"\Fin.scm")
    time.sleep(sleep_time)
    window1[u'開啟(&O)Button'].click()

    '''確認是否已經按下執行鍵運行 Macro 檔'''
    check_run = 1
    while check_run > 0:
        try:
            w_handle1 = pywinauto.findwindows.find_windows(title=u'Select macro file to Load/Execute', class_name="#32770")[0]
            window1 = app.window(handle=w_handle1)
            window1[u'開啟(&O)Button'].click()
        except IndexError:
            print("已開啟 Macro 開始模擬")
            check_run = -1

    '''檢查 start.OML 並在模擬結束後重命名為 completion_signal.OML'''
    find_status = 0
    while find_status == 0:
        if os.path.exists(signal_file_path):  
            OK_or_NOT = 1
            find_status = -1
        
            

    '''模擬結束後進行初始化操作'''
    if OK_or_NOT == 1:
        print("模擬完成，初始化")
        afxe.menu_item(u'&Macros->&Execute').select()
        w_handle1 = pywinauto.findwindows.find_windows(title=u'Select macro file to Load/Execute', class_name="#32770")[0]
        window1 = app.window(handle=w_handle1)
        window1[u'檔案名稱(&N):Edit'].type_keys(r"" + path + "\Data\Reset.scm")
        time.sleep(sleep_time)
        window1[u'開啟(&O)Button'].click()
    else:
        print("模擬未完成或失敗，跳過初始化")

    return OK_or_NOT

tracepro(1)
