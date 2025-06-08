# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 13:58:46 2020

@author: ck101

本檔案為演算法運行前置作業，準備光學模擬步驟儲存資料的資料夾，以及所需的 Macro 檔案
Macro 的原檔存於 Data 資料夾內，複製到 GA_population 資料夾中 P1 ~ PX 資料夾 (X=population)
"""
import os
import shutil

def mkdir(path):
    ''' 判斷目錄是否存在，如果不存在，則建立新目錄'''
    folder = os.path.exists(path)
    
    if not folder:
        os.makedirs(path)
        print('-----建立成功-----')
    else:
        print(path+'目錄已存在')

path = os.getcwd()
txtDir = path + "\\Macro\\" 

'''複製 Macro 檔至 GA_population 資料夾'''
'''高緯度 Final_run.scm'''
#with open(txtDir+"Final_run.scm", "r") as f:
'''低緯度 Fin.scm'''
with open(txtDir+"Sim.scm", "r") as f:
    #old_folder = txtDir + "Final_run.scm"
    old_folder = txtDir + "Sim.scm"
    population = 40
    for i in range(population):
        new_folder = path + "\\GA_population\\P" + str(i+1)
        mkdir(new_folder)
        shutil.copy(old_folder,new_folder)