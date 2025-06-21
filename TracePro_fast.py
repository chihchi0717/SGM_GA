from pywinauto import application, findwindows
import time, os

import warnings

warnings.filterwarnings("ignore", message=".*32-bit application should be automated.*")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
signal = os.path.join(BASE_DIR, "Macro", "completion_signal.OML")
reset_path = os.path.join(BASE_DIR, "Macro", "Reset.scm")
sim_path = os.path.join(BASE_DIR, "Macro", "Sim.scm")
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
    afxc = app["Afx:400000:8:10003:0:bf0aa1"].wait("ready")
    # afxc.menu_item(u'&Window->&1 Model:[initiate.OML]').select()
    afxc.menu_select("&Window->&1 Model:[initiate.OML]")

    app.window(title_re=".*TracePro.*").menu_item("&Macros->&Execute").select()
    w_handle = findwindows.find_windows(title="Select macro file to Load/Execute")[0]
    macro_win = app.window(handle=w_handle)
    macro_win["檔案名稱(&N):Edit"].set_edit_text(path_macro)
    macro_win["開啟(&O)Button"].click()


def tracepro_fast(path_macro, timeout=60, exe_path=None):

    if os.path.exists(signal):
        os.remove(signal)

    if exe_path is None:
        exe_path = (
            r"C:\Program Files (x86)\Lambda Research Corporation\TracePro\TracePro.exe"
        )
    app = application.Application().connect(path=exe_path)
    win = app.window(title_re=".*TracePro.*")
    win.wait("ready")
    time.sleep(0.1)

    # 開啟 Polar 與 Rectangular Candela Plot
    # win.menu_select("&Analysis->Candela Plots->Polar Candela Distribution")
    # time.sleep(0.1)
    # win.menu_select("&Analysis->Candela Plots->Rectangular Candela Distribution")
    # time.sleep(0.1)

    while True:
        # 開啟 Polar 與 Rectangular Candela Plot
        win.menu_select("&Analysis->Candela Plots->Polar Candela Distribution")
        time.sleep(0.1)
        win.menu_select("&Analysis->Candela Plots->Rectangular Candela Distribution")
        time.sleep(0.1)
        print("▶️ 執行模擬 macro:", os.path.basename(path_macro))
        load_macro(app, path_macro)

        if wait_file(signal, timeout=timeout):
            # print("✅ 模擬完成，執行 Reset.scm")
            load_macro(app, reset_path)
            return True

        # 如果到這裡代表超時
        print("⏰ 模擬超時，執行 Reset.scm 並重試")
        load_macro(app, reset_path)
        time.sleep(5)
        win["開啟(&O)Button"].click()
        # 清掉重試前可能殘留的 signal
        # if os.path.exists(signal):
        #     os.remove(signal)
        # 休息一下再重試（可依需要調整）
        time.sleep(1)
    # 執行主 macro
    # load_macro(app, path_macro)
    # start_time = time.time()

    # if wait_file(signal):
    #     #print("模擬完成")
    #     time.sleep(0.1)
    #     load_macro(app, reset_path)
    #     #print("已執行 Reset.scm 初始化")
    # else:
    #     print("超時未完成")
    #     load_macro(app, reset_path)
    # #print("Tracepro Execution time:", round(time.time() - start_time, 2), "sec")
    # return os.path.exists(signal)


# 執行
# tracepro_fast(r"C:\Users\user\Desktop\NTHU\MasterThesis\GA\SGM_GA\Data\Sim.scm")
