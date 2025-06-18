# AutoCAD prism structure generator (stair and triangle modes)
# 加入底座 Base 的版本

import math
import time
import sys
import warnings
from pyautocad import Autocad, APoint
import comtypes
import os

sys.stdout.reconfigure(encoding="utf-8")
warnings.simplefilter("ignore", UserWarning)


def retry_autocad_call(func, retries=3, wait_time=5):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            print(f"AutoCAD call failed, retry {attempt + 1}: {e}")
            if attempt < retries - 1:
                time.sleep(wait_time)
            else:
                raise


def send_command_with_retry(acad, command, retries=5, delay=2):
    for attempt in range(retries):
        try:
            acad.ActiveDocument.SendCommand(command)
            break
        except comtypes.COMError as e:
            print(f"Failed to send command to AutoCAD (attempt {attempt + 1}): {e}")
            time.sleep(delay)
    else:
        raise RuntimeError(f"Failed to execute command after retries: {command}")


def Build_model(sid_ang, mode="stair", folder="."):
    """
    sid_ang: [side_a, side_b, angle_B]
    mode: "stair" 或 "triangle"
    folder: 輸出 .SAT/.DWG 的資料夾
    """

    # --- 參數設定 ----------------------------------------------------
    sat_path = os.path.join(folder, "prism_sat_file-print_sub1.SAT")
    dwg_path = os.path.join(folder, "Drawing.dwg")
    center_y_path = os.path.join(folder, "center_y.txt")
    center_x_path = os.path.join(folder, "center_x.txt")

    start_time = time.time()
    sleep_time = 0.2
    scale = 1  # 1 個「單位座標」代表 0.001 公尺 (根據原本範例)

    side_a, side_b, angle_B = sid_ang
    # 計算三角形頂點 A, B, C
    angle_B2_rad = math.radians(90 - angle_B)
    B = (0, side_a)
    Cx = side_b * math.cos(angle_B2_rad)
    Cy = side_a - side_b * math.sin(angle_B2_rad)
    C = (Cx, Cy)
    A = (0, 0)

    # 三角形第三邊長
    side_c = math.sqrt(
        side_a**2 + side_b**2 - 2 * side_a * side_b * math.cos(math.radians(angle_B))
    )
    # 質心 (僅用來做 BOUNDARY 快速定位)
    Ix = (side_b * A[0] + side_c * B[0] + side_a * Cx) / (side_a + side_b + side_c)
    Iy = (side_b * A[1] + side_c * B[1] + side_a * Cy) / (side_a + side_b + side_c)

    # 直線方程式：AC 與 BC
    slope_ac = Cy / Cx if Cx != 0 else 0
    slope_bc = (Cy - B[1]) / (Cx - B[0]) if (Cx - B[0]) != 0 else 0
    intercept_bc = B[1] - slope_bc * B[0]
    equ_ac = lambda x: slope_ac * x
    equ_bc = lambda x: slope_bc * x + intercept_bc

    pixel_size = 22  # 原來用來把座標對齊到 22 的倍數

    # 啟動 AutoCAD
    acad = retry_autocad_call(lambda: Autocad(create_if_not_exists=True))
    retry_autocad_call(lambda: acad.app.Documents.Add())
    time.sleep(sleep_time)

    # 設定單位為公尺 (2 = 英制英吋 → 4 = 小數米制；以下選「1 單位 = 米」
    try:
        acad.ActiveDocument.SendCommand("-UNITS\n2\n4\n1\n4\n0\nY\n\n")
        time.sleep(sleep_time)
    except Exception as e:
        print(f"⚠️ 設定 UNITS 命令失敗：{e}")
        raise
    send_command_with_retry(acad, "FACETRES\n10\n")

    # --------------- 畫 Prism 主體 (stair 或 triangle) ----------------------
    if mode == "triangle":
        # 計算三角形上下界
        top = (math.floor(equ_bc(0) / pixel_size)) * pixel_size
        bottom = (math.ceil(equ_ac(0) / pixel_size)) * pixel_size
        # 畫三角形三條邊
        for p1, p2 in zip([A, B, C], [B, C, A]):
            acad.model.AddLine(
                APoint(p1[0] * scale, p1[1] * scale),
                APoint(p2[0] * scale, p2[1] * scale),
            )
        center = ((A[0] + B[0] + C[0]) / 3, (A[1] + B[1] + C[1]) / 3)
        # 等待後續 BOUNDARY 與 EXTRUDE...

    elif mode == "stair":
        # 計算 stair 形狀的上下界
        top = (math.floor(equ_bc(0) / pixel_size)) * pixel_size
        bottom = (math.ceil(equ_ac(0) / pixel_size)) * pixel_size
        current_x1 = current_x2 = 0
        current_y1, current_y2 = bottom, top

        pos, pos1, pos2 = [(-22, bottom), (-22, top)], [], []
        # 用來累積 stair 的折線座標
        while True:
            # 計算下一個階梯點
            equ_y1 = equ_ac(current_x1)
            equ_y2 = equ_bc(current_x2)
            real_y1 = math.ceil(equ_y1 / pixel_size) * pixel_size
            real_y2 = math.floor(equ_y2 / pixel_size) * pixel_size

            # 如果 new Y 與 current 不同，加入折線節點
            if real_y2 != current_y2 and current_y1 != real_y2:
                current_x2 -= pixel_size
                pos.append((current_x2, current_y2))
                pos.append((current_x2, real_y2))
                pos2.extend([(current_x2, current_y2), (current_x2, real_y2)])
            elif real_y2 == real_y1:
                break
            else:
                pos.append((current_x2 - pixel_size, real_y2))
                pos.append((current_x2, real_y2))
                pos2.extend([(current_x2 - pixel_size, real_y2), (current_x2, real_y2)])

            if real_y1 != current_y1 and current_y1 != real_y2:
                current_x1 -= pixel_size
                pos.append((current_x1, current_y1))
                pos.append((current_x1, real_y1))
                pos1.extend([(current_x1, current_y1), (current_x1, real_y1)])
            elif real_y1 == real_y2:
                break
            else:
                pos.append((current_x1 - pixel_size, real_y1))
                pos.append((current_x1, real_y1))
                pos1.extend([(current_x1 - pixel_size, real_y1), (current_x1, real_y1)])

            # 更新座標，用於下一輪迭代
            if current_x1 <= current_x2:
                current_x1 += pixel_size
            else:
                current_x1 -= pixel_size
            current_x2 += pixel_size
            current_y1 = real_y1
            current_y2 = real_y2

        # 根據折線座標分別畫兩條 stair
        for seq in [pos1, pos2]:
            for i in range(len(seq) - 1):
                if seq[i] != seq[i + 1]:
                    acad.model.AddLine(
                        APoint(seq[i][0] * scale, seq[i][1] * scale),
                        APoint(seq[i + 1][0] * scale, seq[i + 1][1] * scale),
                    )

        # 畫最左側與最右側的垂直線
        acad.model.AddLine(
            APoint(-22 * scale, bottom * scale), APoint(-22 * scale, top * scale)
        )
        acad.model.AddLine(
            APoint(pos1[-1][0] * scale, pos1[-1][1] * scale),
            APoint(pos2[-1][0] * scale, pos2[-1][1] * scale),
        )

    else:
        raise ValueError("mode must be 'stair' or 'triangle'")

    # ----------------- 對所有邊線做 JOIN → EXTRUDE → UNION -------------------
    send_command_with_retry(acad, "SELECT\nALL\n\n_JOIN\n\n")
    send_command_with_retry(acad, "ZOOM\nE\n\n")
    send_command_with_retry(
        acad, f"-BOUNDARY\n{round(Ix * scale, 4)},{round(Iy * scale, 4)}\n\n"
    )
    send_command_with_retry(acad, "_EXTRUDE\nL\n\n50\n")
    send_command_with_retry(acad, "UNION\nALL\n\n")

    # ----------------- 建立陣列 (30 列、1 欄) -------------------
    base_length_x = 10  # <── 把這裡換成你實際想要的 X 長度
    base_length_y = 55  # <── 把這裡換成你實際想要的 Y 寬度
    base_thickness = 15  # 底座厚度 5 mm (0.005 m)，可依列印機最小極限改
    r = base_length_y / (sid_ang[0] * scale)  # 列數，根據 sid_ang[0] 的實際長度計算
    print(f"列數：{r}")
    rows, columns = int(r), 1
    # row_spacing 使用「實際高度」＋「實際厚度」，這裡用 sid_ang[0] 的實際長度
    row_spacing = (sid_ang[0]) * scale * (rows - 1)
    column_spacing = 1  # 1 米間距 (看需求可自行調整)
    send_command_with_retry(
        acad,
        f"ARRAY\nALL\n\nR\nCOL\n{columns}\nT\n{column_spacing}\nR\n{rows}\nT\n{row_spacing}\n0\nX\n",
    )
    time.sleep(sleep_time)

    # 把陣列拆開、再 UNION 變成單一實體
    send_command_with_retry(acad, "Explode\nALL\n\n")
    send_command_with_retry(acad, "UNION\nALL\n\n")

    # ---------------- 此處新增「基板」 --------------------
    start_base = APoint(0, 0, 0)  # APoint(-Cx, 0, 0)
    sub_length_x = 1.1  # <── 把這裡換成你實際想要的 X 長度
    sub_length_y = 55  # <── 把這裡換成你實際想要的 Y 寬度
    sub_thickness = 50
    send_command_with_retry(  # + Cx* scale
        acad,
        f"_BOX\n{start_base.x - sub_length_x},{start_base.y},{start_base.z}\n{start_base.x},{start_base.y + sub_length_y},{start_base.z + sub_thickness}\n",
    )
    # ---------------- 此處新增「底座 Base」 --------------------
    base_length_x = 10  # <── 把這裡換成你實際想要的 X 長度
    base_length_y = 55  # <── 把這裡換成你實際想要的 Y 寬度
    base_thickness = 15  # 底座厚度 5 mm (0.005 m)，可依列印機最小極限改

    start_base = APoint(0, 0, 0)
    # acad.model.AddBox(start_base, base_length_x, base_length_y, base_thickness)
    send_command_with_retry(
        acad,
        f"_BOX\n{start_base.x - base_length_x},{start_base.y},{start_base.z}\n{start_base.x + Cx* scale},{start_base.y + base_length_y},{start_base.z - base_thickness}\n",
    )

    # ---------------- 儲存 SAT/DWG 然後關閉 --------------------
    send_command_with_retry(acad, f"Export\n{sat_path}\ny\nALL\n\n")
    send_command_with_retry(acad, f"save\n{dwg_path}\ny\n")
    # try:
    #     send_command_with_retry(acad, "close\n")
    #     time.sleep(2)
    # except Exception as e:
    #     print(f"⚠️ 嘗試關閉 AutoCAD 檔案時出錯：{e}")

    # print("Autocad Execution time:", round(time.time() - start_time, 2), "sec")
    return 1


# 範例呼叫：（可自行修改 sid_ang 與模式）
s = 1  # mm
sid_ang = [0.48 * s, 0.98 * s, 85]
Build_model(
    sid_ang,
    mode="triangle",
    folder=r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\202506\0618",
)
