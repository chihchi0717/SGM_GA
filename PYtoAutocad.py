# AutoCAD prism structure generator (stair and triangle modes)
# Optimized implementation with fixed box position at array center

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
    
    # 設定檔名
    sat_path = os.path.join(folder, "prism_sat_file-sim.SAT")
    dwg_path = os.path.join(folder, "Drawing.dwg")
    center_y_path = os.path.join(folder, "center_y.txt")
    center_x_path = os.path.join(folder, "center_x.txt")

    # ===== 在繪圖前先刪除前一代 SAT 檔案 =====
    log = []  # 紀錄訊息

    if os.path.exists(sat_path):
        try:
            os.remove(sat_path)
            log.append(f"🗑️ 已刪除上一代 SAT 檔案: {sat_path}")
        except Exception as e:
            log.append(f"⚠️ 無法刪除舊 SAT 檔案: {sat_path}, 錯誤: {e}")

    if os.path.exists(dwg_path):
        try:
            os.remove(dwg_path)
            log.append(f"🗑️ 已刪除上一代 DWG 檔案: {dwg_path}")
        except Exception as e:
            log.append(f"⚠️ 無法刪除舊 DWG 檔案: {dwg_path}, 錯誤: {e}")
            
    start_time = time.time()
    sleep_time = 0.2
    scale = 1 #0.001 #(mm)

    side_a, side_b, angle_B = sid_ang
    angle_B2_rad = math.radians(90 - angle_B)
    B = (0, side_a)
    Cx = round(side_b * math.cos(angle_B2_rad), 2)
    Cy = round(side_a - side_b * math.sin(angle_B2_rad), 2)
    C = (Cx, Cy)
    A = (0, 0)

    side_c = math.sqrt(
        side_a**2 + side_b**2 - 2 * side_a * side_b * math.cos(math.radians(angle_B))
    )
    if side_a + side_b + side_c == 0:
        print(f"⚠️ Invalid triangle: side_a={side_a}, side_b={side_b}, side_c={side_c}. Skip drawing.")
        return 0  # 或者其他方式：raise Exception / 直接退出本函数

    Ix = round((side_b * A[0] + side_c * B[0] + side_a * Cx) / (side_a + side_b + side_c), 1)
    Iy = round((side_b * A[1] + side_c * B[1] + side_a * Cy) / (side_a + side_b + side_c), 1)

    slope_ac = Cy / Cx
    slope_bc = (Cy - B[1]) / (Cx - B[0])
    intercept_bc = B[1] - slope_bc * B[0]

    equ_ac = lambda x: slope_ac * x
    equ_bc = lambda x: slope_bc * x + intercept_bc

    pixel_size = 22
    

    acad = retry_autocad_call(lambda: Autocad(create_if_not_exists=True))
    #acad.app.Documents.Add()
    retry_autocad_call(lambda: acad.app.Documents.Add())
    while acad.ActiveDocument is None:
        time.sleep(1)

    time.sleep(sleep_time)

    # 設定單位
    # try:
    #     acad.ActiveDocument.SendCommand("-UNITS\n2\n4\n1\n4\n0\nY\n\n")
    #     time.sleep(sleep_time)
    # except Exception as e:
    #     print(f"⚠️ 設定 UNITS 命令失敗：{e}")
    #     raise

    if mode == "triangle":
        # top = sid_ang[0]
        # bottom = 0
        top = equ_bc(0) 
        bottom = equ_ac(0)
        for p1, p2 in zip([A, B, C], [B, C, A]):
            acad.model.AddLine(
                APoint(p1[0] * scale, p1[1] * scale),
                APoint(p2[0] * scale, p2[1] * scale),
            )
        center = ((A[0] + B[0] + C[0]) / 3, (A[1] + B[1] + C[1]) / 3)
        # send_command_with_retry(
        #     acad,
        #     f"-BOUNDARY\n{round(center[0]*scale, 4)},{round(center[1]*scale, 4)}\n\n",
        # )
        
        # send_command_with_retry(acad, "_EXTRUDE\nL\n\n1\n")
        #send_command_with_retry(acad, "UNION\nALL\n\n")

    elif mode == "stair":
        top = (math.floor(equ_bc(0) / pixel_size)) * pixel_size
        bottom = (math.ceil(equ_ac(0) / pixel_size)) * pixel_size
        current_x1 = current_x2 = 0
        current_y1, current_y2 = bottom, top

        pos, pos1, pos2 = [(-22, bottom), (-22, top)], [], []
        step1 = step2 = 0

        while current_y2 > current_y1 and current_x2 <= Cx and current_x1 <= Cx:
            equ_y1 = equ_ac(current_x1)
            equ_y2 = equ_bc(current_x2)
            real_y1 = math.ceil(equ_y1 / pixel_size) * pixel_size
            real_y2 = math.floor(equ_y2 / pixel_size) * pixel_size

            if real_y2 != current_y2 and current_y1 != real_y2:
                current_x2 -= pixel_size
                pos.append((current_x2, current_y2))
                pos.append((current_x2, real_y2))
                pos2.extend([(current_x2, current_y2), (current_x2, real_y2)])
            elif real_y2 == real_y1:
                break
            else:
                step2 += 1
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
                step1 += 1
                pos.append((current_x1 - pixel_size, real_y1))
                pos.append((current_x1, real_y1))
                pos1.extend([(current_x1 - pixel_size, real_y1), (current_x1, real_y1)])

            current_x1 += pixel_size if current_x1 <= current_x2 else -pixel_size
            current_x2 += pixel_size
            current_y1 = real_y1
            current_y2 = real_y2

        for seq in [pos1, pos2]:
            for i in range(len(seq) - 1):
                if seq[i] != seq[i + 1]:
                    acad.model.AddLine(
                        APoint(seq[i][0] * scale, seq[i][1] * scale),
                        APoint(seq[i + 1][0] * scale, seq[i + 1][1] * scale),
                    )

        acad.model.AddLine(
            APoint(-22 * scale, bottom * scale), APoint(-22 * scale, top * scale)
        )
        acad.model.AddLine(
            APoint(pos1[-1][0] * scale, pos1[-1][1] * scale),
            APoint(pos2[-1][0] * scale, pos2[-1][1] * scale),
        )

    else:
        raise ValueError("mode must be 'stair' or 'triangle'")

    send_command_with_retry(acad, "SELECT\nALL\n\n_JOIN\n\n")
    send_command_with_retry(acad, "ZOOM\nE\n\n")
    # send_command_with_retry(
    #     acad, f"-BOUNDARY\n{round(Ix*scale,1)},{round(Iy*scale,1)}\n\n"
    # )
    try:
        send_command_with_retry(acad, f"-BOUNDARY\n{Ix},{Iy}\n\n")
    except Exception as e:
        print(f"⚠️ Build_model: Boundary 失敗 ({e})，跳過 Extrude。")
        return 0
    send_command_with_retry(acad, "_EXTRUDE\nL\n\n1\n")
    time.sleep(0.2)
    #send_command_with_retry(acad, "-BLOCK\nprism\n0,0,0\nL\n\n")

    send_command_with_retry(acad, "UNION\nALL\n\n")

    rows, columns = 30, 1
    #row_spacing = (top - bottom) * scale * (rows - 1)
    row_spacing = (sid_ang[0]) * scale * (rows - 1)
    column_spacing = 1
    send_command_with_retry(
        acad,
        f"ARRAY\nALL\n\nR\nCOL\n{columns}\nT\n{column_spacing}\nR\n{rows}\nT\n{row_spacing}\n0\nX\n",
    )
    

    time.sleep(sleep_time)
    send_command_with_retry(acad, "Explode\nALL\n\n")
    send_command_with_retry(acad, "UNION\nALL\n\n")
    
    light_source_length = 0.5
    actual_array_top = top + (rows - 1) * (top - bottom)
    array_center_y = (actual_array_top + bottom) / 2
    # === Step: 儲存中心 y 座標 ===
    center_y = round(array_center_y * scale, 1)
    
    # with open(r"C:\Users\user\Desktop\NTHU\MasterThesis\GA\SGM_GA\file\center_y.txt", "w") as f:
    #     f.write(str(center_y))

    center_x =round(Cx * scale + 1, 1)# 0 
    # with open(r"C:\Users\user\Desktop\NTHU\MasterThesis\GA\SGM_GA\file\center_x.txt", "w") as f:
    #     f.write(str(center_x))
    # 儲存 center_y 和 center_x
    with open(center_y_path, "w") as f:
        f.write(str(center_y))
    with open(center_x_path, "w") as f:
        f.write(str(center_x))

    #array_center_y = sid_ang[0]/2
    #print("TracePro rotation center:", 30, array_center_y, 0.5)

    start_point = APoint(100, array_center_y * scale, 0)
    send_command_with_retry(
        acad,
        f"_BOX\n{start_point.x},{start_point.y},{start_point.z}\n{start_point.x + light_source_length + 0.6},{start_point.y + light_source_length + 0.6},{start_point.z + light_source_length}\n",
    )

    sat_file_path = (sat_path)
    
    dwg_file_path = (dwg_path)

    # === 儲存 DWG 與 SAT 檔案 ===
    try:
        send_command_with_retry(acad, f"save\n{dwg_path}\ny\n")
    except Exception as e:
        print(f"⚠️ 儲存 DWG 時出錯：{e}")

    try:
        send_command_with_retry(acad, f"Export\n{sat_path}\ny\nALL\n\n")
        time.sleep(2)
    except Exception as e:
        print(f"⚠️ 嘗試儲存 SAT 檔案時出錯：{e}")

    # === 確認 SAT 和 DWG 檔案是否已正確輸出，若失敗則最多重試 3 次 ===
    for retry in range(3):
        if os.path.exists(sat_path) and os.path.exists(dwg_path):
            break  # ✅ 已成功儲存
        print(f"⚠️ 檔案未成功產生，進行第 {retry + 1} 次重試存檔...")
        try:
            send_command_with_retry(acad, f"save\n{dwg_path}\ny\n")
        except Exception as e:
            print(f"⚠️ 重試儲存 DWG 時出錯：{e}")
        try:
            send_command_with_retry(acad, f"Export\n{sat_path}\ny\nALL\n\n")
            time.sleep(2)
        except Exception as e:
            print(f"⚠️ 重試儲存 SAT 時出錯：{e}")
        time.sleep(1)

    # === 最後關閉檔案（只在檔案已生成時才關） ===
    if os.path.exists(sat_path) and os.path.exists(dwg_path):
        try:
            send_command_with_retry(acad, "close\n")
            time.sleep(2)
        except Exception as e:
            print(f"⚠️ 嘗試關閉 AutoCAD 檔案時出錯：{e}")
    else:
        print(f"❌ 最終仍未成功產生檔案：{sat_path} 或 {dwg_path}。")



    #print("Autocad Execution time:", round(time.time() - start_time, 2), "sec")
    return 1, log



#Example usage
# sid_ang = [0.9, 0.9, 64]
# Build_model(sid_ang, 1, "triangle", "C:\Users\123\SGM_GA\GA_population")