# GA_N.py 修改版：只保留三角形流程 + 簡化建模流程
import numpy as np
import TracePro_fast as TracePro
import txt
import draw_New  # 修改為只處理三角形

# ==== 基本參數設定 ====
polygon = 3  # 三角形
DNA_SIZE = 3  # 一邊 + 一角
ANGLE_BOUND = [1, 179]
SIDE_BOUND = [400, 900]
CROSS_RATE = 0.6
MUTATE_RATE = 0.1
POP_SIZE = 100
N_GENERATIONS = 100

class GA:
    def __init__(self):
        self.pop_size = POP_SIZE
        self.changeable_side = 1
        self.changeable_angle = 1
        self.DNA_size = DNA_SIZE
        self.pop = []

    def generate(self):
        pop = []
        attempts = 0
        while len(pop) < self.pop_size:
            attempts += 1
            dna = [np.random.randint(*SIDE_BOUND), np.random.randint(*SIDE_BOUND), np.random.randint(*ANGLE_BOUND)]
            success, _, _ = draw_New.draw_(dna, 1, 1, -1, -1, 0, 1)
            if not success:
                print(f"❌ draw_() 失敗: {dna}")
            else:
                pop.append(dna)
            if attempts > 500:
                print(f"⚠️ 無法產生足夠個體，目前僅 {len(pop)} 個，請檢查 draw_ 判斷條件")
                break
        self.pop = np.array(pop)



    def get_fitness(self, generation):
        fitness_list = []
        for i, dna in enumerate(self.pop):
            success, x, y = draw_New.draw_(dna, 1, 1, generation, i, 0, 0)
            #success, x, y = draw_New.draw_(dna, 1, 1, generation, i, 0, 1)

            if not success:
                print(f"❌ P{i+1}: draw_() failed for dna={dna}")
                fitness_list.append(0.01)
                continue

            try:
                TracePro.tracepro(i)
                fitness, *_ = txt.score_data(i)
            except:
                fitness = 0.01
            fitness_list.append(fitness)
            print(f"Gen {generation+1} - P{i+1}: fitness = {fitness:.3f}")
        return fitness_list

    def select(self, fitness_list):
        probs = np.array(fitness_list) / np.sum(fitness_list)
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, p=probs)
        return self.pop[idx]

    def crossover(self, parent, pop):
        if np.random.rand() < CROSS_RATE:
            mate = pop[np.random.randint(self.pop_size)]
            mask = np.random.randint(0, 2, self.DNA_size).astype(bool)
            parent = np.array(parent)
            mate = np.array(mate)
            parent[mask] = mate[mask]
            return parent.tolist()
        return parent


    def mutate(self, child):
        for i in range(self.DNA_size):
            if np.random.rand() < MUTATE_RATE:
                if i == 0:
                    child[i] = np.random.randint(*SIDE_BOUND)
                elif i == 1:
                    child[i] = np.random.randint(*SIDE_BOUND)
                else:
                    child[i] = np.random.randint(*ANGLE_BOUND)
        return child

    def evolve(self, fitness_list):
        selected = self.select(fitness_list)
        next_gen = []
        for p in selected:
            c = self.crossover(p.copy(), selected)
            c = self.mutate(c)
            next_gen.append(c)
        self.pop = np.array(next_gen)


if __name__ == '__main__':
    ga = GA()
    
    ga.generate()

    x_, y_ = [], []
    for g in range(N_GENERATIONS):
        fitness = ga.get_fitness(g)
        best = max(fitness)
        print(f"=== Generation {g+1} Best: {best:.4f} ===")
        x_.append(g+1)
        y_.append(best)
        ga.evolve(fitness)

    print("Final generation best:", max(y_))