# -*- coding: utf-8 -*-
"""
Created on Thu May 10 15:18:16 2018

@author: pig84
"""
import pygame
import pandas as pd
import math
import time
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def read_files(file_name):
    return pd.read_table(file_name, delimiter  = ' ', header = None, keep_default_na = False).values        

HEIGHT = 600
WIDTH = 450
SCALE = 6
X_ZERO = WIDTH/2
Y_ZERO = HEIGHT/2 + 150

#colors
WHITE = ((255, 255, 255))
RED = ((255, 0, 0))
BLACK = ((0, 0, 0))

#init
def init_pygame():
    pygame.init()
    global gameDisplay
    gameDisplay = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('GUI')
    global clock
    clock = pygame.time.Clock()

def fi_next(fi, theta):
    output = fi - math.degrees(math.asin(2*math.sin(math.radians(theta))/6))
    if output > 270:
        output -= 360
    elif output <= -90:
        output +=360
    #print('fi : ', output, math.degrees(math.asin(2*math.sin(math.radians(theta))/6)))
    return output

def x_next(x, fi, theta):
    output = x + math.cos(math.radians(fi + theta)) + math.sin(math.radians(theta)) * math.sin(math.radians(fi))
    #print('x : ', output)
    return output

def y_next(y, fi, theta):
    output = y + math.sin(math.radians(fi + theta)) - math.sin(math.radians(theta)) * math.cos(math.radians(fi))
    #print('y : ',output)
    #print('-----------')
    return output

def message_display(is_end):
    myfont = pygame.font.SysFont('arial', 30)
    if is_end:
        textsurface = myfont.render('Arrived !!!', True, BLACK)
    else:
        textsurface = myfont.render('Press Enter to start.', True, BLACK)
    textrect = textsurface.get_rect()
    textrect.center = (220, 50)
    gameDisplay.blit(textsurface, textrect)
    
def check_in_end(position, data_1, data_2):
    if position[0]>= data_1[0] and position[0] <= data_2[0] and position[1]>= data_2[1] and position[1] <= data_1[1]:
        return True
    else:
        return False
    
def intersection(x1, y1, a1, x2, y2, a2):
    b1 = y1 - a1*x1
    b2 = y2 - a2*x2
    
    if a1 > 1e4 or a1 < -1e4:
        x = x1
        y = a2*x + b2
    elif a2 > 1e4 or a2 < -1e4:
        x = x2
        y = a1*x + b1
    elif a1 == 0:
        y = y1
        x = (y - b2)/a2
    elif a2 == 0:
        y = y2
        x = (y - b1)/a1
    else:
        x = round((b2 - b1)/(a1 - a2), 2)
        y = a1*x + b1
    #print(x1, y1, a1, x2, y2, a2, 'point :', (x, y))
    return (x, y)

def check_in_segment(point, wall):
    if point[0] <= max(wall[0][0], wall[1][0]) and point[0] >= min(wall[0][0], wall[1][0]) and point[1] <= max(wall[0][1], wall[1][1]) and point[1] >= min(wall[0][1], wall[1][1]):
        return True
    else:
        return False

def check_same_direction(sensor_vector, inter_point, position):
    #print('check_same_direction', sensor_vector, inter_point, position)
    vector = (inter_point[0] - position[0], inter_point[1] - position[1])
    if vector[0]*sensor_vector[0] > 0 or vector[1]*sensor_vector[1] > 0:
        #print('True')
        return True
    else:
        #print('False')
        return False

def calcu_distance(position, point):
    return math.sqrt((position[0] - point[0])**2 + (position[1] - point[1])**2)

class Car(pygame.sprite.Sprite):
    def __init__(self, init_value):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((6*SCALE, 6*SCALE))
        self.rect = self.image.get_rect()
        self.pos_x = int(init_value[0])
        self.pos_y = int(init_value[1])
        self.fi = int(init_value[2])
        self.rect.center = (self.pos_x + X_ZERO, self.pos_y + Y_ZERO)
        self.image.fill(WHITE)
        pygame.draw.circle(self.image, RED, (int(3*SCALE), int(3*SCALE)), 3*SCALE, 3) 
        pygame.draw.line(self.image, RED, (int(3*SCALE), int(3*SCALE)),
                        (3*SCALE*math.cos(math.radians(self.fi)) + int(3*SCALE), -3*SCALE*math.sin(math.radians(self.fi))+ int(3*SCALE)), 3) 
    
    def update(self, theta):
        #render update
        self.image.fill(WHITE)
        pygame.draw.circle(self.image, RED, (int(3*SCALE), int(3*SCALE)), 3*SCALE, 3) 
        pygame.draw.line(self.image, RED, (int(3*SCALE), int(3*SCALE)),
                        (3*SCALE*math.cos(math.radians(self.fi)) + int(3*SCALE), -3*SCALE*math.sin(math.radians(self.fi))+ int(3*SCALE)), 3) 
        
        #parameter update
        self.pos_x = x_next(self.pos_x, self.fi, theta)
        self.pos_y = y_next(self.pos_y, self.fi, theta)
        self.fi = fi_next(self.fi, theta)
        self.rect.center = (self.pos_x*SCALE + X_ZERO, -self.pos_y*SCALE + Y_ZERO)
        
    def get_x(self):
        return self.pos_x
    
    def get_y(self):
        return self.pos_y
    
    def get_fi(self):
        return self.fi
    
    def sensor(self, walls, direction):
        #slope
        if direction == 'front':
            sensor_slope = round(math.tan(math.radians(self.fi)), 4)
            sensor_vector = (round(math.cos(math.radians(self.fi)), 4), round(math.sin(math.radians(self.fi)), 4))
        elif direction == 'right':
            sensor_slope = round(math.tan(math.radians(self.fi - 45)), 4)
            sensor_vector = (round(math.cos(math.radians(self.fi - 45)), 4), round(math.sin(math.radians(self.fi - 45)), 4))
        elif direction == 'left':
            sensor_slope = round(math.tan(math.radians(self.fi + 45)), 4)
            sensor_vector = (round(math.cos(math.radians(self.fi + 45)), 4), round(math.sin(math.radians(self.fi + 45)), 4))
        saved_walls = []
        
        for wall in walls:
            #calculate slope
            wall_slope = 1e10
            if wall.get_points()[0][0] - wall.get_points()[1][0] != 0:
                wall_slope = (wall.get_points()[0][1] - wall.get_points()[1][1]) / (wall.get_points()[0][0] - wall.get_points()[1][0])
            
            #calculate intersection
            inter_point = intersection(self.pos_x, self.pos_y, sensor_slope, wall.get_points()[0][0], wall.get_points()[0][1], wall_slope)
            if check_in_segment(inter_point, wall.get_points()):
                if check_same_direction(sensor_vector, inter_point, (self.pos_x, self.pos_y)):
                    saved_walls.append(calcu_distance((self.pos_x, self.pos_y), inter_point))
        #print('Walls distance :', saved_walls)
        return min(saved_walls)

def collision(x, y, walls):
    collision = False
    for wall in walls:
        #calculate slope
        wall_slope = 1e10
        if wall.get_points()[0][0] - wall.get_points()[1][0] != 0:
            wall_slope = (wall.get_points()[0][1] - wall.get_points()[1][1]) / (wall.get_points()[0][0] - wall.get_points()[1][0])
        
        #calculate intersection
        inter_point = intersection(x, y, 1e-10, wall.get_points()[0][0], wall.get_points()[0][1], wall_slope)
        if check_in_segment(inter_point, wall.get_points()):
            if calcu_distance((x, y), inter_point) < 3:
                collision = True
        #calculate intersection
        inter_point = intersection(x, y, 1e10, wall.get_points()[0][0], wall.get_points()[0][1], wall_slope)
        if check_in_segment(inter_point, wall.get_points()):
            if calcu_distance((x, y), inter_point) < 3:
                collision = True
        #calculate intersection
        inter_point = intersection(x, y, 1, wall.get_points()[0][0], wall.get_points()[0][1], wall_slope)
        if check_in_segment(inter_point, wall.get_points()):
            if calcu_distance((x, y), inter_point) < 3:
                collision = True
        #calculate intersection
        inter_point = intersection(x, y, -1, wall.get_points()[0][0], wall.get_points()[0][1], wall_slope)
        if check_in_segment(inter_point, wall.get_points()):
            if calcu_distance((x, y), inter_point) < 3:
                collision = True
        #calculate intersection
        inter_point = intersection(x, y, (1/2), wall.get_points()[0][0], wall.get_points()[0][1], wall_slope)
        if check_in_segment(inter_point, wall.get_points()):
            if calcu_distance((x, y), inter_point) < 3:
                collision = True
        #calculate intersection
        inter_point = intersection(x, y, -(1/2), wall.get_points()[0][0], wall.get_points()[0][1], wall_slope)
        if check_in_segment(inter_point, wall.get_points()):
            if calcu_distance((x, y), inter_point) < 3:
                collision = True
        #calculate intersection
        inter_point = intersection(x, y, 1.7, wall.get_points()[0][0], wall.get_points()[0][1], wall_slope)
        if check_in_segment(inter_point, wall.get_points()):
            if calcu_distance((x, y), inter_point) < 3:
                collision = True
        #calculate intersection
        inter_point = intersection(x, y, -1.7, wall.get_points()[0][0], wall.get_points()[0][1], wall_slope)
        if check_in_segment(inter_point, wall.get_points()):
            if calcu_distance((x, y), inter_point) < 3:
                collision = True
    return collision
        
class Wall(pygame.sprite.Sprite):
    def __init__(self, start_value, stop_value):
        pygame.sprite.Sprite.__init__(self)
        self.point_1 = start_value
        self.point_2 = stop_value
        
        #vertical
        if (start_value[0] == stop_value[0]):
            self.image = pygame.Surface((1, abs(start_value[1] - stop_value[1])*SCALE))
            self.rect = self.image.get_rect()
            x_mean = (start_value[0] + stop_value[0])/2*SCALE
            y_mean = (start_value[1] + stop_value[1])/2*SCALE
            self.rect.center = (x_mean + X_ZERO, -y_mean + Y_ZERO)
            
            #pygame.draw.line(self.image, RED, (start_value[0]*SCALE, start_value[1]*SCALE), (stop_value[0]*SCALE, stop_value[1]*SCALE), 3)
        #horizon
        elif (start_value[1] == stop_value[1]):
            self.image = pygame.Surface(((abs(start_value[0] - stop_value[0]) + 1)*SCALE, 1))
            self.rect = self.image.get_rect()
            x_mean = (start_value[0] + stop_value[0])/2*SCALE
            y_mean = (start_value[1] + stop_value[1])/2*SCALE
            self.rect.center = (x_mean + X_ZERO, -y_mean + Y_ZERO)
            
            #pygame.draw.line(self.image, RED, (start_value[0]*SCALE, start_value[1]*SCALE), (stop_value[0]*SCALE, stop_value[1]*SCALE), 3)
        else:
            self.image = pygame.Surface((abs(start_value[0] - stop_value[0])*SCALE, 1))
            #self.image.fill(WHITE)
            self.rect = self.image.get_rect()
            x_mean = (start_value[0] + stop_value[0])/2*SCALE
            y_mean = ((start_value[1] + stop_value[1])/2)*SCALE
            self.rect.center = (x_mean + X_ZERO, -y_mean + Y_ZERO)
            pygame.draw.line(self.image, RED, (start_value[0]*SCALE, start_value[1]*SCALE), (stop_value[0]*SCALE, stop_value[1]*SCALE), 3)

    def get_points(self):
        return (self.point_1, self.point_2)
    
def GUI(data, rbfn, scaler):
    n, d = data.shape
    #sprites
    all_sprites = pygame.sprite.Group()
    walls = pygame.sprite.Group()
    car = Car(data[0])
    all_sprites.add(car)
    path = []
    for i in range(3, n-1):
        wall = Wall(data[i], data[i+1])
        all_sprites.add(wall)
        walls.add(wall)
    
    #game loop
    running = True
    start = False
    while running:
        #FPS
        clock.tick(20)
        #get sensor
        front_distance = car.sensor(walls.sprites(), 'front')
        right_distance = car.sensor(walls.sprites(), 'right')
        left_distance = car.sensor(walls.sprites(), 'left')
        
        #process input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    start = True
        #update
        if start:
            sensor_value = np.array([[front_distance, right_distance, left_distance, 1]])
            sensor_value = scaler.transform(sensor_value)
            theta = rbfn.get_theta(sensor_value[:, 0:3], scaler)
            all_sprites.update(theta)
        #collision
        #hits = pygame.sprite.spritecollide(car, walls, False)
        hits = collision(car.get_x(), car.get_y(), walls.sprites())
        if hits:
            running = False
        #end
        is_end = check_in_end((car.get_x(), car.get_y()), data[1], data[2])
        
        #render
        gameDisplay.fill(WHITE)
        message_display(is_end)
        if is_end:
            running = False
        all_sprites.draw(gameDisplay)
        #print path
        path.append((car.get_x(), car.get_y()))
        for i in range(len(path)):
            pygame.draw.circle(gameDisplay, RED, (int(path[i][0]*SCALE + X_ZERO), int(-path[i][1]*SCALE + Y_ZERO)), 1, 1)
        pygame.display.update()
    time.sleep(1)
    pygame.quit()
    quit()


def PSO(rbfn_4d_list, iteration, colony_size, fi_1, fi_2):
    #initial
    best_parameter = []
    best_score = []
    for rbfn in rbfn_4d_list:
        best_parameter.append(rbfn.get_vector())
        best_score.append(1/(rbfn.adaptation_function(rbfn.basis_function())))
    
    for iters in range(iteration):
        #best so far
        for i in range(len(rbfn_4d_list)):
            if best_score[i] < (1/(rbfn_4d_list[i].adaptation_function(rbfn_4d_list[i].basis_function()))):
                best_parameter[i] = rbfn_4d_list[i].get_vector()
        
            #global best
            global_best = rbfn_4d_list[i].get_vector()
            for j in range(len(rbfn_4d_list)):
                if j != i:
                    if rbfn_4d_list[j]
        
class RBFN():
    def __init__(self, data, j):
        self.data = data[:, 0:-1]
        self.truth = data[:, -1:]
        self.data_n, self.data_d = self.data.shape
        self.j = j

        self.mean = 2 * np.random.random((self.j, self.data_d)) - 1
        self.variance = np.random.random((self.j,))
        self.weights = 2 * np.random.random((self.j,)) - 1
        self.theta = 2 * np.random.random((1,)) - 1
        self.function_output = np.zeros((self.data_n, self.j))
        
    def basis_function(self):
        for j_count in range(self.j):
            for i in range(self.data_n):
                _ = (-1 * np.sum(np.square(self.data[i, :] - self.mean[j_count, :])))/ (2 * np.square(self.variance[j_count]))
                self.function_output[i, j_count] = math.exp(_)
                
        return self.function_output.dot(self.weights) + self.theta
    
    def get_theta(self, sensor_value, scaler):
        output = np.zeros((1, self.j))
        for j_count in range(self.j):
            _ = (-1 * np.sum(np.square(sensor_value - self.mean[j_count, :])))/ (2 * np.square(self.variance[j_count]))
            #print('_', _)
            output[:, j_count] = math.exp(_)
        #print(output)
        output = (output.dot(self.weights) + self.theta).reshape(-1, 1)
        
        output = scaler.inverse_transform(np.concatenate((sensor_value, output), axis = 1))
        return output[:, -1]
    
    def adaptation_function(self, fx):
        fx = fx.reshape(-1, 1)
        return np.sum(np.square(self.truth - fx))/2
    
    def get_vector(self):
        return np.concatenate((self.theta, self.weights, self.mean.flatten(), self.variance), axis = 0)
    
    def set_vector(self, next_parameter_vector):
        
        for i in range(len(next_parameter_vector)):
            if i>= self.j*self.data_d + self.j+1:
                if next_parameter_vector[i] > 1 :
                    next_parameter_vector[i] = 1
                elif next_parameter_vector[i] < 0:
                    next_parameter_vector[i] = 10e-8
            elif next_parameter_vector[i] > 1 :
                next_parameter_vector[i] = 1
            elif next_parameter_vector[i] < -1:
                next_parameter_vector[i] = -1
        
        #next_parameter_vector = np.tanh(next_parameter_vector)
        self.theta = np.asarray(next_parameter_vector[0:1])
        self.weights = np.asarray(next_parameter_vector[1 : self.j+1])
        self.mean = np.asarray(next_parameter_vector[self.j+1 : self.j*self.data_d + self.j+1]).reshape(self.j, self.data_d)
        self.variance = np.asarray(next_parameter_vector[self.j*self.data_d + self.j+1 :])
        
        #print(self.theta.shape, self.weights.shape, self.mean.shape, self.variance.shape)
    
def main():
    
    iteration = int(input('Iteration count(>0) :'))
    while (iteration <= 0):
        iteration = int(input('Iteration count(>0) :'))
    colony_size = int(input('Colony size(>0) :'))
    while (colony_size <= 0):
        colony_size = int(input('Colony size(>0) :'))
    fi_1 = float(input('fi_1 (0~1) :'))
    while(fi_1 < 0 or fi_1 > 1):
        fi_1 = float(input('fi_1 (0~1) :'))
    fi_2 = float(input('fi_2 (0~1) :'))
    while(fi_2 < 0 or fi_2 > 1):
        fi_2 = float(input('fi_2 (0~1) :'))
    j = int(input('J(>0) :'))
    while(j <= 0):
        j = int(input('J(>0) :'))
    
    #read data
    data_4d = read_files('./train4D.txt')
    
    #scale data
    scaler4d = MinMaxScaler(feature_range=(-1, 1))
    scaler4d.fit([[0, 0, 0, -40], [80, 80, 80, 40]])
    data_4d = scaler4d.transform(data_4d)
    
    ###RBFN###
    rbfn_4d_list = []
    
    #initalize parameters
    for i in range(colony_size):
        rbfn_4d_list.append(RBFN(data_4d, j))
    
    #training
    print('Training 4D data...')
    start_time = time.time()
    
    '''
    score_list = []
    parameter_vector = []
    #calculate score
    for rbfn in rbfn_4d_list:
        fx = rbfn.basis_function()
        error = rbfn.adaptation_function(fx)
        #print(error)
        error = math.pow(error, 10)
        score_list.append(1/error)
        parameter_vector.append(rbfn.get_vector())
    '''
       
    ###PSO###
    next_parameter_vector = PSO(rbfn_4d_list, iteration, colony_size, fi_1, fi_2)
    
    #print(next_parameter_vector)
    for i in range(len(rbfn_4d_list)):
        rbfn_4d_list[i].set_vector(next_parameter_vector[i])
    
    print('4D Training time :', round((time.time() - start_time), 2), 'sec.')
        
    
    
    
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
    
    ###################pygame###################
    init_pygame()
    df = pd.read_table('./case01.txt', delimiter  = ',', header = None, keep_default_na = False)
    case01_list = df.values
    GUI(case01_list, best_4d_rbfn, scaler4d)
        
    '''
    
    for data in data_4d:
        print(best_4d_rbfn.get_theta(data[0:3].reshape(1, -1), scaler4d))
    '''
if __name__ == '__main__':
    main()