import time
import os
from pywinauto import application, findwindows

def wait_file(path, timeout=60):
    start = time.time()
    while not os.path.exists(path):
        if time.time() - start > timeout:
            return False
        time.sleep(0.2)
    return True

def load_macro(app, path_macro):
    """打開並執行指定的 .scm 檔案"""
    app.TracePro.menu_item(u'&Macros->&Execute').select()
    w_handle = findwindows.find_windows(title=u'Select macro file to Load/Execute')[0]
    macro_win = app.window(handle=w_handle)
    macro_win[u'檔案名稱(&N):Edit'].set_edit_text(path_macro)
    macro_win[u'開啟(&O)Button'].click()

def tracepro_fast(path_macro):
    signal = r"C:\Users\user\Desktop\NTHU\MasterThesis\GA\SGM_GA\file\completion_signal.OML"
    reset_path = r"C:\Users\user\Desktop\NTHU\MasterThesis\GA\SGM_GA\Data\Reset.scm"

    if os.path.exists(signal):
        os.remove(signal)

    app = application.Application().connect(path="C:\\Program Files (x86)\\Lambda Research Corporation\\TracePro\\TracePro.exe")
    win = app.TracePro
    win.wait('ready')

    # 執行主 macro
    load_macro(app, path_macro)
    start_time = time.time()

    if wait_file(signal):
        print("模擬完成")

        # 模擬完成後執行 Reset
        time.sleep(1)  # 等待 TracePro 穩定
        load_macro(app, reset_path)
        print("已執行 Reset.scm 初始化")

    else:
        print("超時未完成")

    print("Execution time:", round(time.time() - start_time, 2), "sec")
    return os.path.exists(signal)

# 測試執行
tracepro_fast(r"C:\Users\user\Desktop\NTHU\MasterThesis\GA\SGM_GA\Data\Sim.scm")
