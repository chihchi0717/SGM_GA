# main.py：整合 GA 流程（產生參數、建模、模擬、評估、選擇、交配）

import random
import os
import numpy as np
from draw_New import draw_
from PYtoAutocad_New0523_light_center_short import Build_model
from TracePro_fast import tracepro_fast
import txt  # 用於計算 fitness

# ==== GA 超參數設定 ====
POP_SIZE = 10
N_GENERATIONS = 5
SIDE_BOUND = [400, 900]
ANGLE_BOUND = [1, 179]
CROSS_RATE = 0.6
MUTATE_RATE = 0.1


def generate_valid_population(n):
    population = []
    attempts = 0
    while len(population) < n and attempts < n * 10:
        a = random.randint(*SIDE_BOUND)
        b = random.randint(*SIDE_BOUND)
        A = random.randint(*ANGLE_BOUND)
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
    return np.array(population)


def get_fitness(pop, generation):
    fitness_list = []
    for i, dna in enumerate(pop):
        success, *_ = draw_(dna, 1, 1, generation, i, 0, 0)
        if not success:
            print(f"❌ P{i+1}: draw_() failed for dna={dna}")
            fitness_list.append(0.01)
            continue

        try:
            tracepro_fast(os.path.abspath("./Data/Sim.scm"))
            fitness, *_ = txt.score_data(i)
        except Exception as e:
            print(f"⚠️ 模擬錯誤: {e}")
            fitness = 0.01

        print(f"Gen {generation+1} - P{i+1}: fitness = {fitness:.3f}")
        fitness_list.append(fitness)
    return np.array(fitness_list)


def select(pop, fitness):
    probs = fitness / fitness.sum()
    idx = np.random.choice(np.arange(len(pop)), size=len(pop), p=probs)
    return pop[idx]


def crossover(p1, p2):
    mask = np.random.randint(0, 2, len(p1)).astype(bool)
    child = np.where(mask, p1, p2)
    return child


def mutate(child):
    for i in range(len(child)):
        if np.random.rand() < MUTATE_RATE:
            if i < 2:
                child[i] = random.randint(*SIDE_BOUND)
            else:
                child[i] = random.randint(*ANGLE_BOUND)
    return child


# ==== 主程式 ====
if __name__ == "__main__":
    pop = generate_valid_population(POP_SIZE)

    for g in range(N_GENERATIONS):
       
        print(f"\n=== Generation {g+1} ===")
        for i, individual in enumerate(pop):
            print(f"建模個體 P{i+1}: {individual}")
            
            # 設定當前個體的工作資料夾
            folder_path = f"./GA_population/P{i+1}"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # 切換當前工作目錄
            os.chdir(folder_path)

            # 建模並導出 .sat 檔等
            Build_model(individual, mode="triangle")

            # 切回原始路徑
            os.chdir("../../")


        fitness = get_fitness(pop, g)
        best = np.max(fitness)
        print(f"=== Generation {g+1} Best: {best:.4f} ===")

        selected = select(pop, fitness)
        next_gen = []

        for i in range(POP_SIZE):
            p1 = selected[np.random.randint(POP_SIZE)]
            p2 = selected[np.random.randint(POP_SIZE)]
            child = crossover(p1.copy(), p2.copy())
            child = mutate(child)
            next_gen.append(child)

        pop = np.array(next_gen)

    print("✅ GA 完成")
