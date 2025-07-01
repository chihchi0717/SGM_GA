# -*- coding: utf-8 -*-
# The setting on MSI desktop

import serial
import time
import re
import pandas as pd
import openpyxl
import argparse
from datetime import datetime

# === 可設定參數 ===
STEP_PER_DEGREE = 400  # 每度所需脈衝數
DEGREE_STEP = 3  # 每次旋轉角度

ELEVATION_ANGLE = 10  # 設定仰角
PRISM_ANGLE = 90 - ELEVATION_ANGLE
now_str = datetime.today().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILENAME = f"./{now_str}_0.6_1.2_79_center_ele{ELEVATION_ANGLE}.xlsx"


# 初始化控制器串口連線（SHOT-602）
controller_serial = serial.Serial(
    port="COM3",  # 替換為控制器的串口號
    baudrate=9600,  # 與控制器的波特率匹配
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=0.5,
    xonxoff=False,
    rtscts=True,
)

# 初始化 P-LINK 串口連線
plink_serial = serial.Serial(
    port="COM4",  # 替換為 P-LINK 的串口號
    baudrate=57600,  # 根據手冊設定
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=1,
)

# 確認串口連接
if controller_serial.is_open and plink_serial.is_open:
    print(f"控制器已連接到: {controller_serial.portstr}")
    print(f"P-LINK 已連接到: {plink_serial.portstr}")


# 控制器相關函數
def send_command(command):
    """向控制器發送命令並獲取回應"""
    command += "\r\n"
    controller_serial.write(command.encode("ascii"))
    time.sleep(0.5)
    response = controller_serial.read_all().decode("ascii").strip()
    return response


def wait_until_stop(timeout=10):
    """
    等待馬達停止，避免尚未停穩就進行下一步。
    這裡以定時輪詢 Q: 指令為例，假設回傳包含 BUSY / READY 狀態。
    實際依照手冊做調整。
    """
    start_time = time.time()
    while (time.time() - start_time) < timeout:
        controller_serial.write(b"Q:\r\n")
        resp = controller_serial.readline().decode("ascii").strip()
        # print(f"等待停止: {resp}")
        # 根據實際手冊判斷回傳格式，例如: '1:R' 代表軸 1 Ready
        if "R" in resp:
            return True
        time.sleep(0.2)
    print("警告：等待停止逾時")
    return False


def get_current_pulse(axis):
    """
    透過 Q: 指令，取得目前脈衝位置 (假設回傳格式類似: "3600,         0,K,K,B")
    回傳整數脈衝值。
    """
    controller_serial.write(b"Q:\r\n")
    resp = controller_serial.readline().decode("ascii").strip()
    # 假設字串類似 "3600,         0,K,K,B"
    # 我們以逗號分割
    parts = resp.split(",")
    if len(parts) < 1:
        print(f"Q:回傳內容無法拆解: {resp}")
        return 0

    # 第一欄位為脈衝值
    # pulse_str = parts[0].strip() # e.g. "3600"
    if axis == "1":
        pulse_str = int(parts[0].strip().replace(" ", ""))
        print(f"脈衝值: {pulse_str}")
    else:
        pulse_str = int(parts[1].strip().replace(" ", ""))
        print(f"脈衝值: {pulse_str}")

    return pulse_str


# 讀取初始脈衝
def init_pulse(axis):
    global home_pulse_position

    home_pulse_position = get_current_pulse(axis)
    print(f"軸 {axis} ，目前脈衝: {home_pulse_position}")

    controller_serial.write(b"Q:\r\n")
    response = controller_serial.readline().decode("ascii").strip()
    print(f"Status response: {response}")

    stop_stage()


def init_stage(axis):
    """
    僅做初始化動作，例如歸零馬達、校正之類。
    """
    print("執行初始化流程，不做量測...")

    if axis == "1":
        print("初始化下層平台...")
        init_results_1 = []
        angle_resolution = 0.2
        range_degree = 1
        move_stage(
            axis="1", direction="+", degrees=range_degree
        )  # move to -1 * lower limit
        time.sleep(0.5)  # 等待穩定
        measure_energy()

        total_steps = int(range_degree / angle_resolution) * 2
        print(f"總步數: {total_steps} 步")
        for step in range(1, total_steps + 1):

            move_stage(axis="1", direction="-", degrees=angle_resolution)  # 旋轉控制器
            time.sleep(1)  # 等待穩定
            stop_stage()  # 停止控制器
            energy = measure_energy()

            current_pulse = get_current_pulse(1)

            init_results_1.append(
                {
                    "init_angle": step * angle_resolution,
                    "init_energy": energy,
                    "pulse": current_pulse,
                }
            )

            print(
                f"角度: {step * angle_resolution} 度, 測量能量: {energy}, 脈衝: {current_pulse}"
            )

        print(init_results_1)

        if init_results_1:
            max_item = max(init_results_1, key=lambda x: x["init_energy"])
            print("\n=== 最大能量 ===")
            print(f"能量: {max_item['init_energy']}")
            print(f"對應角度: {max_item['init_angle']} 度")
            print(f"對應脈衝: {max_item['pulse']}")

            max_pulse_position = max_item["pulse"]
            difference = current_pulse - max_pulse_position

            if difference == 0:
                print("已在max位置，無需移動")
                return

            # 根據差值的正負，決定移動方向
            if difference > 0:
                # 目前脈衝大於 home => 要往負方向移回
                move_stage(
                    axis, direction="-", degrees=abs(difference / STEP_PER_DEGREE)
                )
            else:
                # 目前脈衝小於 home => 要往正方向移回
                move_stage(
                    axis, direction="+", degrees=abs(difference / STEP_PER_DEGREE)
                )

    elif axis == "2":

        print("初始化上層平台...")
        init_results_2 = []
        angle_resolution = 0.5
        range_degree = 1
        move_stage(axis="2", direction="+", degrees=range_degree)  # move to -10
        time.sleep(0.5)  # 等待穩定
        measure_energy()

        total_steps = int(1 / range_degree) * 2
        for step in range(1, total_steps + 1):

            move_stage(axis="2", direction="-", degrees=angle_resolution)  # 旋轉控制器
            time.sleep(1)  # 等待穩定
            stop_stage()  # 停止控制器
            energy = measure_energy()

            current_pulse = get_current_pulse(2)

            init_results_2.append(
                {
                    "init_angle": step * angle_resolution,
                    "init_energy": energy,
                    "pulse": current_pulse,
                }
            )

            print(
                f"角度: {step * angle_resolution} 度, 測量能量: {energy}, 脈衝: {current_pulse}"
            )

        print(init_results_2)

        if init_results_2:
            max_item = max(init_results_2, key=lambda x: x["init_energy"])
            print("\n=== 最大能量 ===")
            print(f"能量: {max_item['init_energy']}")
            print(f"對應角度: {max_item['init_angle']} 度")
            print(f"對應脈衝: {max_item['pulse']}")

            max_pulse_position = max_item["pulse"]
            difference = current_pulse - max_pulse_position

            if difference == 0:
                print("已在max位置，無需移動")
                return

            # 根據差值的正負，決定移動方向
            if difference > 0:
                # 目前脈衝大於 home => 要往負方向移回
                move_stage(
                    axis, direction="-", degrees=abs(difference / STEP_PER_DEGREE)
                )
            else:
                # 目前脈衝小於 home => 要往正方向移回
                move_stage(
                    axis, direction="+", degrees=abs(difference / STEP_PER_DEGREE)
                )

    print("初始化完畢。")


def move_stage(axis, direction, degrees):
    """將控制器旋轉指定角度"""
    pulses = int(degrees * STEP_PER_DEGREE)
    command = f"M:{axis}{direction}P{pulses}\r\n"
    send_command(command)
    send_command("G")  # 啟動移動
    time.sleep(0.1)  # 根據速度調整延遲
    controller_serial.write(b"Q:\r\n")
    response = controller_serial.readline().decode("ascii").strip()
    print(f"Status response: {response}")


def stop_stage():
    """停止控制器運動"""
    send_command("L:W")
    # send_command("1:W")
    # send_command("2:W")


def back2init(axis):
    global home_pulse_position
    current_pulse = get_current_pulse(axis)
    difference = current_pulse - home_pulse_position
    print(
        f"目前脈衝 = {current_pulse}, Home時脈衝 = {home_pulse_position}, 差值 = {difference}"
    )

    if difference == 0:
        print("已在Home位置，無需移動")
        return

    # 根據差值的正負，決定移動方向
    if difference > 0:
        # 目前脈衝大於 home => 要往負方向移回
        move_stage(axis, direction="-", degrees=abs(difference / STEP_PER_DEGREE))
    else:
        # 目前脈衝小於 home => 要往正方向移回
        move_stage(axis, direction="+", degrees=abs(difference / STEP_PER_DEGREE))

    # 等待完成
    wait_until_stop()
    stop_stage()
    print(f"已回到 Home 時的脈衝位置 ({home_pulse_position}).")


# P-LINK 相關函數
def send_plink_command(command):
    """向 P-LINK 發送命令並返回回應"""
    command += "\r"
    plink_serial.write(command.encode("ascii"))
    time.sleep(0.1)
    response = plink_serial.read_all().decode("ascii").strip()
    return response


def wavelength():
    """波長"""
    response = send_plink_command("*PWC0546")
    print(f"wave length response : {response}")
    # response = send_plink_command("*F01")
    # print(f"current wave length : {response}")
    # if not response or "E01" in response:
    #     back2init(axis='1')
    #     raise ValueError("波長設定無效")
    return response


def measure_energy(retry=3, wait=0.5):
    """
    向 P-LINK 讀取能量值，如第一次只回傳 ACK，則等 wait秒後重試。
    retry: 總共嘗試次數
    wait:  每次失敗後等待秒數
    """
    for attempt in range(1, retry + 1):
        response = send_plink_command("*CVU")
        # [DEBUG] 觀察回傳
        print(f"[DEBUG] 第 {attempt} 次量測回傳: {repr(response)}")

        if not response:
            # 回傳空字串 (連ACK都沒有) => 直接重試
            print("[WARN] 回傳是空字串，重試...")
        elif "ACK" in response and not _has_digit(response):
            # 只有 ACK 而無數字 => 可能裝置還沒量完
            print("[WARN] 只收到 ACK，尚未準備好，等待後重試...")
        else:
            # 進入正常解析流程
            if "ACK" not in response:
                # 若裝置要求一定要 "ACK" 代表成功，也可檢查
                print("[INFO] 回傳不含 ACK，但包含數字，嘗試解析...")

            # 做清理
            cleaned = (
                response.replace("\n", "").replace("ACK", "").replace("+", "").strip()
            )
            if not cleaned:
                print("[WARN] 數字部分空字串，重試...")
            else:
                try:
                    energy_val = float(cleaned) * 10**6
                    print(f"[DEBUG] 成功解析: {energy_val}")
                    return energy_val
                except ValueError:
                    print(f"[ERROR] 無法轉成 float: {cleaned}, 重試...")

        # 若還沒成功就等待，然後進下一輪
        time.sleep(wait)

    # 如果多次嘗試都失敗，就拋出錯誤或回傳 None
    raise ValueError(f"測量能量時仍無數字回傳 (嘗試 {retry} 次)")


def _has_digit(s):
    """檢查字串中是否包含 0~9 數字。"""
    return any(ch.isdigit() for ch in s)


def main(elevation_angle, prism_angle, degree_step, output_filename):
    init_pulse(axis="1")
    results = []
    wavelength()
    try:
        total_steps = 180 // degree_step
        time.sleep(5)

        for step in range(total_steps + 1):
            move_stage(axis="1", direction="+", degrees=degree_step)
            time.sleep(1)
            energy = measure_energy()
            time.sleep(1)
            stop_stage()
            print(f"角度: {(step) * degree_step} 度, 測量能量: {energy}")
            results.append({"angle": (step) * degree_step, "energy": energy})

        results_df = pd.DataFrame(results)
        results_df.to_excel(output_filename, index=False)
        print("準備回到 Home 時的脈衝位置...")
        back2init(axis="1")
        time.sleep(2)
        back2init(axis="1")

    except Exception as e:
        back2init(axis="1")
        print(f"發生錯誤: {e}")

    finally:
        if controller_serial.is_open:
            controller_serial.close()
            print("已關閉控制器串口")
        if plink_serial.is_open:
            plink_serial.close()
            print("已關閉 P-LINK 串口")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="控制 SHOT-602 與 P-LINK 的程式")
    parser.add_argument("--init-one", action="store_true", help="初始化軸1")
    parser.add_argument("--init-two", action="store_true", help="初始化軸2")
    parser.add_argument("--ele-ang", action="store_true", help="設定仰角")
    parser.add_argument("--prism-ang", action="store_true", help="旋轉稜鏡到出射角0度")
    parser.add_argument("--move", action="store_true", help="移動")
    parser.add_argument("--measure", action="store_true", help="單次測量")
    parser.add_argument("--stop", action="store_true", help="stop")
    parser.add_argument("--wavelength", action="store_true", help="wavelength")
    args = parser.parse_args()

    if args.init_one:
        init_stage(axis="1")
    elif args.init_two:
        init_stage(axis="2")
    elif args.ele_ang:
        move_stage(axis="2", direction="-", degrees=ELEVATION_ANGLE)
    elif args.prism_ang:
        move_stage(axis="1", direction="-", degrees=PRISM_ANGLE)
    elif args.move:
        move_stage(axis="1", direction="-", degrees=90)
    elif args.measure:
        energy = measure_energy()
        print(f"{energy:.2f} uW")
    elif args.stop:
        stop_stage()
    elif args.wavelength:
        wl = wavelength()
        print(f"{wl}")
    else:
        main(
            elevation_angle=ELEVATION_ANGLE,
            prism_angle=PRISM_ANGLE,
            degree_step=DEGREE_STEP,
            output_filename=OUTPUT_FILENAME,
        )
