# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 13:58:46 2020

@author: ck101

本檔案之函數由 GA_N 檔案呼叫用，其中包含:
1.score_data(): 會讀取 GA_population 資料夾中的 txt 檔案，以定義的適應函數，對光學模擬軟體輸出之資料評分(population_number是第幾個個體)，最後回傳分數。
2.save_data(): 選出前5高分的參數組記錄至 txt，並回傳這5組參數，保留至下一次 generation 迭代用。
"""
import copy 
import time
import os

def score_data(population_number):
    
    '''防止造成眩光的適應函數評分區間: 1.懲罰區間 w1(-68 ~ 0 度)  2.獎勵區間 w2(0 ~ 5 度)  3.獎勵區間 w3(5 ~ 90 度)'''
    w1 = -10
    w2 = 0.1
    w3 = 1
    
    efficient_array = []
    fit_score_array = []
    total_efficient = 0
    eight_fit_efficient = 0
    total_fit_score = 0

    '''高緯度'''
    '''設定不同仰角區間的光線評分後，乘上的權重 weight，5 度 ~ 55 度共11個區間'''
    '''for i in range(11):
        if i == 0:
            weight = 5
        elif i == 1:
            weight = 0.75
        elif i == 2:
            weight = 2.5
        elif i == 3:
            weight = 3.75
        elif i == 4:
            weight = 5.25
        elif i == 5:
            weight = 6.5
        elif i == 6:
            weight = 0.5
        elif i == 7:
            weight = 1.25
        elif i == 8:
            weight = 2
        elif i == 9:
            weight = 2.5
        elif i == 10:
            weight = 1.5'''
    
    '''低緯度'''
    for i in range(8):
        if i == 0:
            weight = 1
        elif i == 1:
            weight = 2
        elif i == 2:
            weight = 5
        elif i == 3:
            weight = 7
        elif i == 4:
            weight = 5
        elif i == 5:
            weight = 8.5
        elif i == 6:
            weight = 1.5
        elif i == 7:
            weight = 2
        data_in = [] # 進入室內光線
        # data_out = [] # 回到室外光線
        
        '''準備讀取檔案的路徑'''
        path = os.getcwd()
        population_folder = path + "\\GA_population\\P" + str(population_number+1)
        time.sleep(0.2)
        #j = (i+1)*5 # 打開 txt 檔案用的名稱，對應名字由 Macro 檔案所定義之存檔名稱
        j = (i+1)*10 # 打開 txt 檔案用的名稱，對應名字由 Macro 檔案所定義之存檔名稱
        with open(population_folder+"\\prism light trace data-4_"+str(j)+".txt","r") as f:
            line_number = 0
            for line in f.readlines():
                '''通過指定分隔符對字串進行切片,返回分割後的字串列表，str.split()分隔符預設為空格，因光學模擬軟體 TracePro 輸出的 txt 檔案中，
                    空格數不一，於是需要先將空兩格修改為空一格、去掉tab鍵，在進行字串分割，換行字符視為不同字串。'''
                line = line.strip()
                line = line.lstrip('\n')
                line = line.replace(' ','')
                line = line.split('\t')
            
                '''txt 前7行為軟體版本、存檔時間等訊息，忽略不儲存'''
                if line_number > 7: 
                    data_in.append(float(line[1])) # 進入室內光線
                    # data_out.append(float(line[2])) # 回到室外光線(本研究暫無用到)
                    
                line_number += 1
                
            '''儲存特定角度區域總能量的變數'''
            score_68_0 = 0 # -68 ~ 0 度(扣 w1 分區域)
            score_0_5 = 0 # 0 ~ 5 度(加 w2 分區域)
            score_5_90 = 0 # 5 ~ 90 度(加 w3 分區域)
            energy_68up = 0 # 0 ~ 5 度(加 w2 分區域)
            energy_up = 0 # -68 ~ 90 度
            total_energy = 0.1 # 防止無任何光線進入室內之情況(防止計算導光效率時分母為 0)

            same = 1 # 防止變數殘留
            for l in range(361):
                '''將重複紀錄的數值消除(測量角度切得太細可能發生能量數值同時存在數個連續角度)'''
                if l>=1:
                    if (data_in[l]==data_in[l-1]) and (data_in[l]!=0):
                        data_in[l] = 0.0
                        same = data_in[l-1]
                    elif data_in[l] == same:
                        data_in[l] = 0.0

                '''接著記錄各角度區域內的光線能量'''

                '''加總所有進入室內光線能量'''
                total_energy += data_in[l]

                '''光線-68以上加總'''
                if l > 44:
                    energy_68up += data_in[l]

                '''光線向上加總'''
                if l > 180:
                    energy_up += data_in[l]

                '''-68 ~ 0 度'''
                if 44 < l < 180:
                    score_68_0 += data_in[l]

                    '''0 ~ 5 度'''
                elif 179 < l < 191:
                    score_0_5 += data_in[l]

                    '''5 ~ 90 度'''
                elif l > 190:
                    score_5_90 += data_in[l]


            '''防止造成眩光的 fitness function，各角度區域內的光線能量比率乘以該區域的權重(w1~w3)'''
            #fit_score = weight*(10 + w1*score_68_0/total_energy + w2*score_0_5/total_energy + w3*score_5_90/total_energy)
            
            '''最大化導光效率的 fitness function，各仰角光線的比重乘以該角度導光效率(被重新導向向上的光線，佔進入室內中所有光線能量的比率)'''
            fit_score =  weight*energy_up/total_energy

            total_fit_score += round(fit_score, 2) # 加總該結構在不同仰角光線的分數
            fit_score_array.append(round(fit_score, 2)) # 該結構稜鏡在各仰角光線的分數
            #eight_fit_efficient += round(fit_score, 2) # 該結構稜鏡累計的分數??????????????????????????
            
            efficient = energy_up/total_energy # 導光效率(被重新導向向上的光線，佔進入室內中所有光線能量的比率)
            total_efficient += round(efficient, 2) # 加總該結構在不同仰角光線的導光效率
            efficient_array.append(round(efficient, 2)) # 該結構稜鏡在各仰角光線的效率

    print("score = ", total_fit_score ) # 適應函數得到總分
    print("total_efficient = ", total_efficient ) # 導光效率總和
    print("efficient_array = ", efficient_array ) # 各角度導光效率
    return total_fit_score, fit_score_array, total_efficient, efficient_array

#score_data(1)

def save_data(population, generation, score_all, score_array, efficient_all, efficient_array ):
    '''        所有參數組、   世代數、   加總分數、  分別分數、    加總效率、        分別效率'''
    path = os.getcwd()
    score_all_copy = copy.deepcopy(score_all) # 找加總分數最高分的參數組
    efficient_all_copy = copy.deepcopy(efficient_all) # 找加總分數最高分參數組的導光效率
    top5_score_array = []
    top5_efficient_array = []
    top5_number = []
    store_next = []

    '''挑出最高分，共進行五次'''
    for _ in range(5):
        max_score_all_copy = max(score_all_copy)
        index = score_all_copy.index(max_score_all_copy)
        score_all_copy[index] = 0
        top5_score_array.append(round(max_score_all_copy, 2)) # 紀錄最高分參數組
        top5_number.append(index+1) # 最高分在第幾組
        top5_efficient_array.append(round(efficient_all_copy[index], 2)) # 紀錄最高分參數組的導光總效率
        store_next.append(population[index])
    
    '''儲存所有參數組的數值與評分'''
    with open(path + "\\history.txt","a") as f:
        for i in range(len(population)): 
            f.write("\nNO."+str(generation + 1)+"-"+str(i + 1)+"data"+str(population[i])+", 總分數 = "+str(score_all[i])+", 各角度效率 = "+str(efficient_array[i]))
        f.write("\nfit前五高分:第"+str(top5_number)+"個，總分數分別為"+str(top5_score_array))
        f.write("\n前五高分之eff:第"+str(top5_number)+"個，總效率分別為"+str(top5_efficient_array))
    
    '''儲存前五高分參數組的數值與評分'''
    with open(path + "\\history_top5.txt","a") as f2: 
        '''儲存前五高分參數組的分數'''    
        for i in range(5):
            f2.write("\nNO."+str(generation + 1)+"-"+str(top5_number[i])+"data"+str(population[top5_number[i]-1])+", 總分數 = "+str(round(score_all[top5_number[i]-1], 2))+", 各角度分數 = "+str(score_array[top5_number[i]-1]))
        f2.write("\nfit前五高分:第"+str(top5_number)+"個，總分數分別為"+str(top5_score_array))

        '''儲存前五高分參數組的導光效率''' 
        for i in range(5):
            f2.write("\nNO."+str(generation + 1)+"-"+str(top5_number[i])+"data"+str(population[top5_number[i]-1])+", 總效率 = "+str(round(efficient_all[top5_number[i]-1], 2))+", 各角度效率 = "+str(efficient_array[top5_number[i]-1]))
        f2.write("\n前五高分之eff:第"+str(top5_number)+"個，總效率分別為"+str(top5_efficient_array))
        f2.write("\n")

    print("store next = ",store_next)
    return store_next