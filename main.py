# 主程式整合流程：產生參數、檢查三角形、AutoCAD 建模、TracePro 模擬

import random
from draw_New import draw_
from PYtoAutocad_New0523_light_center_short import Build_model
from TracePro_fast import tracepro_fast
import os
import time
import math as m

def is_triangle_valid(sides_angles):
    a, b, A_deg = sides_angles
    # 使用餘弦定理驗證是否為三角形
    try:
        A_rad = A_deg * 3.1415926 / 180
        c = (a**2 + b**2 - 2*a*b* m.cos(A_rad))**0.5
        if c <= 0:
            return False
    except:
        return False
    return True


def generate_valid_population(n):
    population = []
    attempts = 0
    while len(population) < n and attempts < n * 10:
        a = random.randint(400, 900)
        b = random.randint(400, 900)
        A = random.randint(10, 170)
        param = [a, b, A]
        try:
            success, *_ = draw_(param, 1, 1, -1, -1, 0, 0)
            if success:
                population.append(param)
                print(f"✅ 有效個體: {param}")
            else:
                print(f"❌ draw_() 失敗: {param}")
        except Exception as e:
            print(f"⚠️ 例外: {param} -> {e}")
        attempts += 1

    if len(population) < n:
        raise RuntimeError(f"無法產生足夠有效個體，僅 {len(population)} 個")
    return population


if __name__ == "__main__":
    population_size = 5
    population = generate_valid_population(population_size)

    for idx, dna in enumerate(population):
        print(f"\n=== 執行個體 {idx+1}: {dna} ===")

        # Step 1. AutoCAD 建模
        Build_model(dna, idx + 1, mode="triangle")

        # Step 2. TracePro 模擬
        sim_macro_path = os.path.abspath("./Data/Sim.scm")
        tracepro_fast(sim_macro_path)

        print(f"✅ 個體 {idx+1} 完成 AutoCAD 建模與 TracePro 模擬")
