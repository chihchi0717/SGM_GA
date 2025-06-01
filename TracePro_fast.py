from pywinauto import application, findwindows
import time, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
signal = os.path.join(BASE_DIR, "Macro", "completion_signal.OML")
reset_path = os.path.join(BASE_DIR, "Macro", "Reset.scm")
# print("== BASE_DIR:", BASE_DIR)
# print("== signal full path:", signal)
# print("== reset_path full path:", reset_path)
# print("== exists signal?", os.path.exists(signal))
# print("== exists reset_path?", os.path.exists(reset_path))

def wait_file(path, timeout=60):
    start = time.time()
    while not os.path.exists(path):
        if time.time() - start > timeout:
            return False
        time.sleep(0.2)
    return True

def load_macro(app, path_macro):
    afxc = app[u'Afx:400000:8:10003:0:bf0aa1'].wait('ready')
    # afxc.menu_item(u'&Window->&1 Model:[initiate.OML]').select()
    afxc.menu_select(u'&Window->&1 Model:[initiate.OML]')
    

    app.window(title_re=".*TracePro.*").menu_item(u'&Macros->&Execute').select()
    w_handle = findwindows.find_windows(title=u'Select macro file to Load/Execute')[0]
    macro_win = app.window(handle=w_handle)
    macro_win[u'檔案名稱(&N):Edit'].set_edit_text(path_macro)
    macro_win[u'開啟(&O)Button'].click()

def tracepro_fast(path_macro):
    
    if os.path.exists(signal):
        os.remove(signal)

    # TRACEPRO_EXE = os.environ.get("TRACEPRO_PATH",
    #     r"C:\Program Files (x86)\Lambda Research Corporation\TracePro\TracePro.exe"
    # )
    #app = application.Application().connect(path=TRACEPRO_EXE)
    app = application.Application().connect(path=r"C:\Program Files (x86)\Lambda Research Corporation\TracePro\TracePro.exe")
    win = app.window(title_re=".*TracePro.*")
    win.wait('ready')
    time.sleep(0.1)

    # 開啟 Polar 與 Rectangular Candela Plot
    win.menu_select("&Analysis->Candela Plots->Polar Candela Distribution")
    time.sleep(0.1)
    win.menu_select("&Analysis->Candela Plots->Rectangular Candela Distribution")
    time.sleep(0.1)
        

    # 執行主 macro
    load_macro(app, path_macro)
    start_time = time.time()

    if wait_file(signal):
        #print("模擬完成")
        time.sleep(0.1)
        load_macro(app, reset_path)
        #print("已執行 Reset.scm 初始化")
    else:
        print("超時未完成")

    print("Tracepro Execution time:", round(time.time() - start_time, 2), "sec")
    return os.path.exists(signal)

# 執行
# tracepro_fast(r"C:\Users\user\Desktop\NTHU\MasterThesis\GA\SGM_GA\Data\Sim.scm")
