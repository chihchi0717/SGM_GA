# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 13:06:50 2020

@author: ck101

本檔案為基因演算法主架構，包括參數生成、交叉、變異，迭代至指定次數，輸出結果。
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
import asyncio
#import screenshot
import PYtoSW_New
import TracePro
import txt
import draw_New
#import cv2
#import time
import os


polygon = 3 # N邊形
#DNA_SIZE = 2*(2*polygon-3)  # 雙邊三角
DNA_SIZE = 2*polygon-3    # 每條染色體有幾個參數
ANGLE_BOUND = [1, 90]    # 角度最大值
SIDE_BOUND = [1, 30]     # 邊長最大值
CROSS_RATE = 0.6         # 交叉機率0.6~1
MUTATE_RATE = 0.1        # 突變機率<0.1  0.0001
POP_SIZE = 100      # 種群數量(N條染色體)
N_GENERATIONS = 100       # 幾個世代 #100

fix_side = 10 # 固定邊的長度
changeable_side = polygon-2
changeable_angle = polygon-2
max_side = 30  # 基因中允許出現的最大邊長值#30
max_angle = 90 # 基因中允許出現的最大角度值
mode = 0 # 1 = 雙結構    0 = 單結構

class GA(object):
    def __init__(self, DNA_size, angle_bound, side_bound, cross_rate, mutation_rate, pop_size, changeable_side, changeable_angle, mode):
        self.DNA_size = DNA_size
        self.angle_bound = angle_bound
        self.side_bound = side_bound
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size
        self.changeable_angle = changeable_angle
        self.changeable_side = changeable_side
        self.mode = mode
        #self.pop = np.random.randint(*DNA_bound, size=(pop_size, DNA_size))

    def generate(self, max_angle, max_side):
        pop = []

        # 讀取舊存檔為1  新模擬為0
        load = 0
        if load == 1:
            '''讀取舊存檔'''
            store_pop_ = []
            path = os.getcwd()
            with open(path + "\\store.txt","r") as f:
                for line in f.readlines():
                    store_pop = []
                    '''整理 txt 檔，去除不必要字符'''
                    line = line.strip() #去除首尾空格
                    line = line.strip('[')
                    line = line.strip(']')
                    line = line.lstrip('\n') #截掉字串左邊的空格或指定字符
                    # line = line.replace(' ','') # replace(old, new[, max]) old:被替換的字串。new:新字串，替換old字串。max:可選字串,替換不超過 max 次
                    line = line.strip()
                    line = line.replace('   ',',') #去掉列表中每一個元素的換行符
                    line = line.replace('  ',',')
                    line = line.replace(' ',',') 
                    line = line.split(',') # 分隔符，默認為所有的空字符，包括空格、換行(\n)、制表符(\t)等。
                    
                    for i in range(len(line)):
                        store_pop.append(int(line[i]))
                        
                    store_pop_.append(store_pop)
                store_pop_ = np.array(store_pop_)
            self.pop = store_pop_ 
            print("讀取舊檔案")   

        else:
            '''進行新參數組生成'''
            pass_num = 0
            while pass_num<self.pop_size:
                temp_ = []
                temp_.append(10) # fix_side  固定垂直邊用
                #temp_.append(np.random.randint(1, 11)) # 垂直邊
                for k in range(self.changeable_side):# 生成邊長
                    temp_.append(np.random.randint(1, max_side+1)) #modify max_side-max_side+1
                for j in range(self.changeable_angle):
                    temp_.append(np.random.randint(1, max_angle+1)) #modify max_angle-max_angle+1
                    # 隨機浮點數random.uniform(min, max)
                if mode == 1:
                    '''複合結構，需要多生成一組參數定義形狀'''
                    #temp.append(10) # fix_side  固定垂直邊用
                    temp_.append(np.random.randint(1, 11)) # 垂直邊
                    for k in range(self.changeable_side):
                        temp_.append(np.random.randint(1, max_side+1)) #modify max_side-max_side+1
                    for j in range(self.changeable_angle):
                        temp_.append(np.random.randint(1, max_angle+1)) #modify max_angle-max_angle+1
                    temp_.append(np.random.randint(0, 2)) # position #?

                success, x_, y_ = draw_New.draw_(temp_, self.changeable_side, self.changeable_angle, -1, -1, self.mode, 1) # 畫圖後判定建模可行性, 1 for debug
                #print((temp_, self.changeable_side, self.changeable_angle, -1, -1, self.mode, 0))
                '''通過判定數量計次(觀察用)'''
                if success == 1:
                    pop.append(temp_)
                    pass_num += 1
                    print("生成數", pass_num, "/", self.pop_size)
            '''生成完成，儲存參數組至 pop 變數'''
            pop = np.array(pop)
            self.pop = pop
        print(self.pop)


    def get_fitness(self, generation, success_num, mode):
        '''紀錄成功建模次數，關閉 SolidWorks 視窗用'''
        success_num_each_round = success_num
        
        all_fitness = []
        sum_fitness = 0
        #all_efficient1 = []
        all_efficient = []
        all_fit_score_array = []
        all_efficient_array = []
        gene_check = []
        
        '''依次評分每個個體(參數組)'''
        for population_number in range(self.pop_size):
            sid_ang = self.pop[population_number] # 儲存第 i 組邊長角度參數
            gene_chec_ = str(self.pop[population_number]) # 紀錄當次參數組 (用於 check 有無重複)
            gene_check.append(str(self.pop[population_number])) # 紀錄已經模擬過的參數組
            success, x__, y__ = draw_New.draw_(sid_ang, self.changeable_side, self.changeable_angle, generation, population_number, mode, 0) # 進行建模可行性判定

            '''判定參數組有無重複，重複就跳過'''
            if success == 1 and population_number > 0:
                if population_number == 1:
                    pop_check = gene_check[0]
                else:
                    pop_check = gene_check[0:population_number-1]

                try:
                    matches = pop_check.index(gene_chec_)
                    print("已在第"+str(1+pop_check.index(gene_chec_))+"個重複計算過了")
                    '''相同參數紀錄相同分數'''
                    all_fit_score_array.append(all_fit_score_array[matches])
                    all_efficient_array.append(all_efficient_array[matches])
                    #all_efficient1.append(all_efficient1[matches])
                    all_efficient.append(all_efficient[matches])
                    all_fitness.append(all_fitness[matches])
                    fitness = all_fitness[matches]
                    success = 2
                except ValueError:
                    success = 1

            '''若判定是新參數組，沒有重複，進行稜鏡建模'''
            if success == 1:
                try:
                    a = asyncio.get_event_loop().run_until_complete(PYtoSW_New.Build_model(x__, y__, success_num_each_round, mode))
                    success_num_each_round += 1

                except KeyboardInterrupt:
                    print("SolidWorks didn't work")
                    sys.exit()
                    

            '''經過建模可行性判定為無法建模，給 0.01 分'''
            if success == 0:
                fitness = 0.01
                all_fit_score_array.append((0,0,0,0,0,0,0,0))
                all_efficient_array.append((0,0,0,0,0,0,0,0))
                #all_efficient1.append(0)
                all_efficient.append(0)
            elif success == 1:
                OK_or_NOT = TracePro.tracepro(population_number) #modify
                #OK_or_NOT = asyncio.get_event_loop().run_until_complete(TracePro.tracepro(population_number))
                print("OK_or_NOT",OK_or_NOT)
                '''OK_or_NOT = 1 光線追蹤模擬成功， = 0 失敗'''
                if OK_or_NOT == 1:
                    #total_fit_score, eight_1_efficient, fit_score_array, total_efficient, efficient_array = txt.score_data(population_number)
                    total_fit_score, fit_score_array, total_efficient, efficient_array = txt.score_data(population_number)

                    fitness = total_fit_score # 計算權重後總分
                    all_fit_score_array.append(fit_score_array) # 計算權重後各仰角分數
                    all_efficient.append(total_efficient) # 導光效率加總
                    all_efficient_array.append(efficient_array) # 各仰角導光效率
                    #all_efficient1.append(eight_1_efficient)

                else:
                    fitness = 0.01
                    all_fit_score_array.append((0,0,0,0,0,0,0,0))
                    all_efficient.append(0)
                    all_efficient_array.append((0,0,0,0,0,0,0,0))
                    #all_efficient1.append(0)
                
            print("generation "+ str(generation+1) + " No " + str(population_number+1) + " fitness = ",fitness)
            sum_fitness += fitness

            if success != 2:
                all_fitness.append(fitness) # 列出各參數組計算權重後總分分數

        if generation == 100:
            self.mutate_rate = 0.1
        #return all_fitness, sum_fitness, all_efficient1, all_fit_score_array, all_efficient, all_efficient_array, success_num_each_round
        '''all_fitness: 列出各參數組計算權重後總分分數
           sum_fitness: 加總各參數組計算權重後總分分數
           all_fit_score_array: 計算權重後各仰角分數
           all_efficient: 導光效率加總
           all_efficient_array: 各仰角導光效率
           success_num_each_round: 建模成功次數
        '''
        return all_fitness, sum_fitness, all_fit_score_array, all_efficient, all_efficient_array, success_num_each_round
        

    def select(self, all_fitness, sum_fitness):
        '''依照分數高低挑選參數組'''
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=np.array(all_fitness)/sum_fitness)
        return self.pop[idx]

    '''modify'''
    def tournament_select(self, all_fitness, sum_fitness):
        selected_parents = []
        while len(selected_parents) < self.pop_size:
            # 隨機選取兩個parent的索引
            parent_indices = np.random.choice(np.arange(len(all_fitness)), size=2)
            # 挑選fitness最高的parent
            best_parent_idx = np.argmax([all_fitness[idx] for idx in parent_indices])
            # 將fitness最高的parent放入被選中的parent陣列
            selected_parents.append(self.pop[parent_indices[best_parent_idx]])
        return np.array(selected_parents)
    '''modify'''

    def crossover(self, parent, pop):
        '''交叉'''
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size, size=1)  # 選pop中的一條基因 [0,pop_size)
            if self.mode == 1:
                length = self.DNA_size*2 + 1
            else:
                length = self.DNA_size
            cross_points = np.random.randint(0, 2, length).astype(np.bool_) # choose crossover points   0~N共N個參數 0=不換 1=替換
            parent[cross_points] = pop[i_, cross_points] # mating and produce one child 
        return parent

    def mutate(self, child,generation):
        '''變異'''
        if self.mode == 1:
            DNA_mutate = self.DNA_size*2+1 # 複合結構
        else:
            DNA_mutate = self.DNA_size # 單一結構
        
       
        if generation<80:
            '''在設定範圍內隨機變化'''
            for point in range(DNA_mutate):
                if np.random.rand() < self.mutate_rate:
                    if point == 0:
                        ####child[point] = 10 #暫用#####
                        child[point] = np.random.randint(1,11)
                    elif point <= self.changeable_side:
                        child[point] = np.random.randint(*self.side_bound)
                    elif point <= (self.changeable_side + self.changeable_angle):
                        child[point] = np.random.randint(*self.angle_bound)
                    elif point == (self.changeable_side + 1 + self.changeable_angle):
                        child[point] = np.random.randint(1,11)
                    elif point <= (self.changeable_side*2 + 1 + self.changeable_angle):
                        child[point] = np.random.randint(*self.side_bound)
                    elif point <= (self.changeable_side*2 + 1 + self.changeable_angle*2):
                        child[point] = np.random.randint(*self.angle_bound)
                    else:
                        child[point] = np.random.randint(0,2)
        else:
            '''在該數值左右小範圍內隨機變化'''
            for point in range(DNA_mutate):
                if np.random.rand() < self.mutate_rate:
                    '''垂直邊變化+-2，角度變化+-10'''
                    # 垂直邊
                    if point == 0:
                        if child[point] + 2 > 10:
                            child[point] = np.random.randint(child[point]-2,11)
                        elif child[point] - 3 < 1:
                            child[point] = np.random.randint(1,child[point]+3)
                        else:
                            child[point] = np.random.randint(child[point]-2,child[point]+3)
                    # 其他邊
                    elif point <= self.changeable_side:
                        if child[point] + 2 > 40:
                            child[point] = np.random.randint(child[point]-2,41)
                        elif child[point] - 3 < 1:
                            child[point] = np.random.randint(1,child[point]+3)
                        else:
                            child[point] = np.random.randint(child[point]-2,child[point]+3)
                    # 其他角
                    else:
                        if child[point] + 10 > 180:
                            child[point] = np.random.randint(child[point]-10,180)
                        elif child[point] - 11 < 1:
                            child[point] = np.random.randint(1,child[point]+11)
                        else:
                            child[point] = np.random.randint(child[point]-10,child[point]+11)    
        return child

    def evolve(self, all_fitness, sum_fitness, top_three, generation,line_y):
        '''進行迭代'''
        path = os.getcwd()
        '''紀錄歷次迭代參數組'''
        with open(path + "\\store.txt","a") as f:
            for line in self.pop:
                f.write(str(line)) # 這句話自帶檔案關閉功能，不需要再寫f.close()
                f.write(str('\n'))
                print(line)
            f.write(str(generation+1))
            f.write(str(line_y))
            f.write(str('\n'))

        pop = self.select(all_fitness, sum_fitness) # 挑選
        #pop = self.tournament_select(all_fitness, sum_fitness) # 挑選

        pop_copy = pop.copy() # 複製
        for parent in pop:
            child = self.crossover(parent, pop_copy) # 交叉
            child = self.mutate(child,generation) # 變異
            parent[:] = child
        self.pop = pop
        '''留下前三高分，存至下一次迭代'''
        for i in range(3):
            self.pop[i] = top_three[i]
        
    def output(self):
        return self.pop

class Line(object):
    '''畫利次迭代中最高分變化圖'''
    def __init__(self, N_GENERATIONS, polygon):
        self.n_generations = N_GENERATIONS
        self.polygon = polygon
        plt.ion()

    def plotting(self, lines_x, lines_y):
        path = os.getcwd()
        plt.cla()
        fig1 = plt.figure(1)
        plt.ylabel('score')
        plt.xlabel('generation')
        plt.plot(lines_x, lines_y) # 各點座標
        plt.xlim((0, self.n_generations)) # X軸範圍-世代
        # plt.ylim((np.min(lines_y), np.max(lines_y))) # Y軸範圍-最佳值
        # plt.ylim((0, 8)) # Y軸範圍-最佳值
        plt.ylim((0, 1+np.max(lines_y))) # Y軸範圍-最佳值
        plt.ioff()
        plt.savefig(path + "\\GA_population\\output.png")
        plt.draw()
        plt.pause(3)
        plt.close(fig1)

import requests
def LineNotify(generation):
    url = 'https://notify-api.line.me/api/notify'
    token = '8nKDVpHKlr4xpzjv0zfGtG6u08bMCFJD09oqKwpIEXp'
    headers = {
        'Authorization': 'Bearer ' + token    # 設定權杖
    }
    data = {
            'message': generation+1     # 設定要發送的訊息
        }
    data = requests.post(url, headers=headers, data=data)   # 使用 POST 方法

if __name__ == '__main__':
    '''設定演算法所需參數'''
    ga = GA(DNA_size = DNA_SIZE, angle_bound = ANGLE_BOUND, side_bound = SIDE_BOUND,
        cross_rate = CROSS_RATE, mutation_rate = MUTATE_RATE, pop_size = POP_SIZE,
        changeable_side = changeable_side, changeable_angle = changeable_angle, mode = mode)
    env = Line(N_GENERATIONS, polygon)
    ga.generate(max_angle, max_side) # 生成參數組
    
    n = 1
    line_x = []
    line_y = []
    success_num = 0
    '''進行迭代'''
    for generation in range(N_GENERATIONS):
        generation_ = generation # 中途斷掉(假設迭代了28次)，把 load 改 1，generation_ = generation + 28
        all_fitness, sum_fitness, all_fit_score_array ,all_efficient, all_efficient_array, success_num_each_round = ga.get_fitness(generation_, success_num, mode) # 計算分數
        LineNotify(generation_)
        OKpop = ga.output()
        top_three = txt.save_data(OKpop, generation_, all_fitness, all_fit_score_array, all_efficient, all_efficient_array)
        line_x.append(n)
        line_y.append(round(np.max(all_fitness), 2))
        ga.evolve(all_fitness, sum_fitness, top_three, generation_,line_y) # 進行迭代
        print('Gen:', generation_+1, '| best fit:', np.max(all_fitness)) # 顯示分數
        n += 1
        if n>1:
            env.plotting(line_x, line_y) # 畫圖
            print(line_y)
        # if generation%2 == 0:
        #     env.plotting(line_x, line_y) # 畫圖
        success_num = success_num_each_round
    output = ga.output()
    print("Final output = ",output)
    # plt.ioff()
    # plt.show()
