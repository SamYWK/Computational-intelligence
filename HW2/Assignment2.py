# -*- coding: utf-8 -*-
"""
Created on Thu May 10 15:18:16 2018

@author: pig84
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def read_files(file_name):
    return pd.read_table(file_name, delimiter  = ' ', header = None, keep_default_na = False).values

class RBFN():
    def __init__(self, data, j):
        self.data = data[:, 0:-1]
        self.truth = data[:, -1:]
        self.data_n, self.data_d = self.data.shape
        self.j = j
        
        self.mean = 80 * np.random.random((self.j, self.data_d))
        self.variance = np.random.random((self.j,))
        self.weights = 2 * np.random.random((self.j,)) - 1
        self.theta = 2 * np.random.random((1,)) - 1
        self.function_output = np.zeros((self.data_n, self.j))
        
    def basis_function(self):
        for j_count in range(self.j):
            for i in range(self.data_n):
                _ = (-1 * np.sum(np.square(self.data[i, :] - self.mean[j_count, :])))/ 2 * np.square(self.variance[j_count])
                self.function_output[i, j_count] = np.exp(_)
                
        return self.function_output.dot(self.weights.T) + self.theta
    
    def adaptation_function(self, fx):
        fx = fx.reshape(-1, 1)
        return np.sum(np.square(self.truth - fx))/2
    
    def get_vector(self):
        #print(np.concatenate((self.theta, self.weights, self.mean.flatten(), self.variance), axis = 0))
        return np.concatenate((self.theta, self.weights, self.mean.flatten(), self.variance), axis = 0)
    
    def set_vector(self, next_parameter_vector):
        
        for i in next_parameter_vector[0:1]:
            if i > 1:
                i = 1
            elif i < -1:
                i = -1
        self.theta = np.asarray(next_parameter_vector[0:1])
        for i in next_parameter_vector[1 : self.j+1]:
            if i > 1:
                i = 1
            elif i < -1:
                i = -1
        self.weights = np.asarray(next_parameter_vector[1 : self.j+1])
        for i in next_parameter_vector[self.j+1 : self.j*self.data_d + self.j+1]:
            if i > 1:
                i = 1
            elif i < -1:
                i = -1
        self.mean = np.asarray(next_parameter_vector[self.j+1 : self.j*self.data_d + self.j+1]).reshape(self.j, self.data_d)
        for i in next_parameter_vector[self.j*self.data_d + self.j+1 :]:
            if i > 1:
                i = 1
            elif i < 0:
                i = 10e-5
        self.variance = np.asarray(next_parameter_vector[self.j*self.data_d + self.j+1 :])
        #print(self.theta.shape, self.weights.shape, self.mean.shape, self.variance.shape)
        
    
def genetic_algo(score_list, colony_size, parameter_vector, crossover_rate, mutation_rate):
    
    ###reproduction###
    reproduction_count = [0] * colony_size
    choice = np.random.choice(2, 1)
    
    #change score into probility
    sum_of_score_list = sum(score_list)
    for i in range(len(score_list)):
        score_list[i] = score_list[i] / sum_of_score_list
        
    #roulette
    if choice == 0:
        chosen_index = (np.random.choice(colony_size, 4, p = score_list))
        for i in chosen_index:
            reproduction_count[i] += 1
    
    #tournament
    else:
        score_list = [i*colony_size for i in score_list]
        reproduction_count = [int(round(i)) for i in score_list]
        
        #check count
        while(sum(reproduction_count) != colony_size):
            if sum(reproduction_count) > colony_size:
                maximum_index = [i for i, j in enumerate(score_list) if j == max(score_list)]
                reproduction_count[maximum_index[0]] -= 1
                #print('-1')
            else :
                minimum_index = [i for i, j in enumerate(score_list) if j == min(score_list)]
                reproduction_count[minimum_index[0]] += 1
                #print('+1')
                
    #print(reproduction_count)
    ###crossover###
    crossover_dice = np.random.random()
    #do crossover
    if crossover_dice < crossover_rate:
        #create pool
        pool = []
        for i in range(colony_size):
            for j in range(reproduction_count[i]):
                pool.append(parameter_vector[i])
        
        for i in range(int(colony_size/2)):
            choice = np.random.choice(2, 1)
            si = np.random.random()
            if choice == 0 :
                temp_1 = pool[i * 2] + si * (pool[i * 2] - pool[i * 2 + 1])
                temp_2 = pool[i * 2 + 1] - si * (pool[i * 2] - pool[i * 2 + 1])
                pool[i * 2] = temp_1
                pool[i * 2 + 1] = temp_2
            else:
                temp_1 = pool[i * 2] + si * (pool[i * 2 + 1] - pool[i * 2])
                temp_2 = pool[i * 2 + 1] - si * (pool[i * 2 + 1] - pool[i * 2])
                pool[i * 2] = temp_1
                pool[i * 2 + 1] = temp_2
    #print(pool)
    ###mutation###
    mutation_dice = np.random.random()
    #do mutation
    if mutation_dice < mutation_rate:
        s = 0.1 * np.random.random()
        choice = np.random.choice(colony_size, 1)
        noise = np.random.random((len(pool[choice[0]]),)) - 0.5
        pool[choice[0]] = pool[choice[0]] + s * noise
    #print(pool)
    return pool
    
def main():    
    interation = int(input('Iteration count(>0) :'))
    while (interation <= 0):
        interation = int(input('Iteration count(>0) :'))
    colony_size = int(input('Colony size(>0) :'))
    while (colony_size <= 0):
        colony_size = int(input('Colony size(>0) :'))
    crossover_rate = float(input('Crossover rate (0~1) :'))
    while(crossover_rate < 0 or crossover_rate > 1):
        crossover_rate = float(input('Crossover rate (0~1) :'))
    mutation_rate = float(input('Mutation rate (0~1) :'))
    while(mutation_rate < 0 or mutation_rate > 1):
        mutation_rate = float(input('Mutation rate (0~1) :'))
    j = int(input('J(>0) :'))
    while(j <= 0):
        j = int(input('J(>0) :'))
    
    #read data
    data_4d = read_files('./train4dAll.txt')
    data_6d = read_files('./train6dAll.txt')
    
    #scale data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit([[0, 0, 0, -40], [80, 80, 80, 40]])
    data_4d = scaler.transform(data_4d)
    scaler.fit([[-75, -75, 0, 0, 0, -40], [75, 75, 80, 80, 80, 40]])
    data_6d = scaler.transform(data_6d)
    
    ###RBFN###
    rbfn_4d_list = []
    rbfn_6d_list = []
    
    #initalize parameters
    for i in range(colony_size):
        rbfn_4d_list.append(RBFN(data_4d, j))
    for i in range(colony_size):
        rbfn_6d_list.append(RBFN(data_6d, j))
    
    #training
    print('Training 4D data...')
    for i in range(interation):
        score_list = []
        parameter_vector = []
        #calculate score
        for rbfn in rbfn_4d_list:
            fx = rbfn.basis_function()
            error = rbfn.adaptation_function(fx)
            #print(error)
            score_list.append(1/error)
            parameter_vector.append(rbfn.get_vector())
           
        ###GA###
        next_parameter_vector = genetic_algo(score_list, colony_size, parameter_vector, crossover_rate, mutation_rate)
        
        for i in range(len(rbfn_4d_list)):
            rbfn_4d_list[i].set_vector(next_parameter_vector[i])
    
    print('Training 6D data...')
    for i in range(interation):
        score_list = []
        parameter_vector = []
        #calculate score
        for rbfn in rbfn_6d_list:
            fx = rbfn.basis_function()
            error = rbfn.adaptation_function(fx)
            print(error)
            score_list.append(1/error)
            parameter_vector.append(rbfn.get_vector())
           
        ###GA###
        next_parameter_vector = genetic_algo(score_list, colony_size, parameter_vector, crossover_rate, mutation_rate)
        
        for i in range(len(rbfn_4d_list)):
            rbfn_6d_list[i].set_vector(next_parameter_vector[i])
    
    #select best parameter
    best_error = 1e8
    best_index = 0
    rbfn_index = 0
    for rbfn in rbfn_4d_list:
        fx = rbfn.basis_function()
        error = rbfn.adaptation_function(fx)
        if error < best_error:
            best_error = error
            best_index = rbfn_index
        rbfn_index += 1
    best_4d_rbfn = rbfn_4d_list[best_index]
    
    best_error = 1e8
    best_index = 0
    rbfn_index = 0
    for rbfn in rbfn_6d_list:
        fx = rbfn.basis_function()
        error = rbfn.adaptation_function(fx)
        if error < best_error:
            best_error = error
            best_index = rbfn_index
        rbfn_index += 1
    best_6d_rbfn = rbfn_6d_list[best_index]
    
    
    
if __name__ == '__main__':
    main()