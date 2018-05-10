# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 16:56:25 2018

@author: pig84
"""

import pygame
import pandas as pd
import math
import time

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
pygame.init()
gameDisplay = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('GUI')
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

def turn_right_small(value):
    #(1/10)x - 1 = y
    #-(1/10)x + 3 = y
    if value == 0:
        return 0
    elif value == 1:
        molecule = 0
        denominator = 0
        for i in range(100, 300):
            j = i/10
            if j <= 20:
                molecule += ((1/10)*j - 1)*j
                denominator += ((1/10)*j - 1)
            else:
                molecule += (-(1/10)*j + 3)*j
                denominator += -(1/10)*j + 3
        return molecule/denominator
    else:
        molecule = 0
        denominator = 0
        for i in range(100, 300):
            j = i/10
            if j <= (value*10 - 1):
                molecule += ((1/10)*j - 1)*j
                denominator += ((1/10)*j - 1)
            elif j > (value*10 - 1) and j <= (value - 3)*(-10):
                molecule += value * j 
                denominator += value
            else:
                molecule += (-(1/10)*j + 3)*j
                denominator += -(1/10)*j + 3
        return molecule/denominator
    
def turn_left_small(value):
    #(1/10)x - 1 = y
    #-(1/10)x + 3 = y
    if value == 0:
        return 0
    elif value == 1:
        molecule = 0
        denominator = 0
        for i in range(100, 300):
            j = i/10
            if j <= 20:
                molecule += ((1/10)*j - 1)*j
                denominator += ((1/10)*j - 1)
            else:
                molecule += (-(1/10)*j + 3)*j
                denominator += -(1/10)*j + 3
        return -(molecule/denominator)
    else:
        molecule = 0
        denominator = 0
        for i in range(100, 300):
            j = i/10
            if j <= (value*10 - 1):
                molecule += ((1/10)*j - 1)*j
                denominator += ((1/10)*j - 1)
            elif j > (value*10 - 1) and j <= (value - 3)*(-10):
                molecule += value * j 
                denominator += value
            else:
                molecule += (-(1/10)*j + 3)*j
                denominator += -(1/10)*j + 3
        return -(molecule/denominator)
    
def turn_right_large(value):
    if value == 0:
        return 0
    elif value == 1:
        molecule = 0
        denominator = 0
        for i in range(300, 400):
            j = i /10
            denominator += 1
            molecule += j
        return molecule/denominator
    else:
        molecule = 0
        denominator = 0
        for i in range(300, 400):
            j = i /10
            denominator += value
            molecule += value * j 
        return molecule/denominator
    
def turn_left_large(value):
    if value == 0:
        return 0
    elif value == 1:
        molecule = 0
        denominator = 0
        for i in range(300, 400):
            j = i /10
            denominator += 1
            molecule += j
        return -(molecule/denominator)
    else:
        molecule = 0
        denominator = 0
        for i in range(300, 400):
            j = i /10
            denominator += value
            molecule += value * j 
        return -(molecule/denominator)

def fuzzy(front_sensor, right_sensor, left_sensor):
    theta = 0
    #antecedent
    front_sensor_flag = []
    front_sensor_value = 0
    right_sensor_flag = []
    right_sensor_value = 0
    left_sensor_flag = []
    left_sensor_value = 0
    #front
    #Small
    if front_sensor <= 10:
        front_sensor_flag.append('small')
        front_sensor_value = 1
    elif front_sensor > 10 and front_sensor <= 12:
        front_sensor_flag.append('small')
        front_sensor_value = -(1/2) * front_sensor + (6)

    #large
    if front_sensor > 12 and front_sensor <= 18:
        front_sensor_flag.append('large')
        front_sensor_value = (1/6) * front_sensor - 2
    elif front_sensor > 18:
        front_sensor_flag.append('large')
        front_sensor_value = 1
        
    ################################################################
    #right
    #Small
    if right_sensor <= 10:
        right_sensor_flag.append('small')
        right_sensor_value = 1
    elif right_sensor > 10 and right_sensor <= 12:
        right_sensor_flag.append('small')
        right_sensor_value = -(1/2) * right_sensor + 6
    #Large
    if right_sensor > 12 and right_sensor <= 20:
        right_sensor_flag.append('large')
        right_sensor_value = (1/8) * right_sensor - (3/2)
    elif right_sensor > 20:
        right_sensor_flag.append('large')
        right_sensor_value = 1
    ################################################################
    #left
    #Small
    if left_sensor <= 10:
        left_sensor_flag.append('small')
        left_sensor_value = 1
    elif left_sensor > 10 and left_sensor <= 12:
        left_sensor_flag.append('small')
        left_sensor_value = -(1/2) * left_sensor + 6
    #Large
    if left_sensor > 12 and left_sensor <= 20:
        left_sensor_flag.append('large')
        left_sensor_value = (1/8) * left_sensor - (3/2)
    elif left_sensor > 20:
        left_sensor_flag.append('large')
        left_sensor_value = 1
        
    #print('F', front_sensor_flag)
    #print('R', right_sensor_flag)
    #print('L', left_sensor_flag)
    
    #conclusion
    if 'large' in right_sensor_flag and 'small' in left_sensor_flag:
        #print('RS:', min(right_sensor_value, left_sensor_value), turn_right_small(min(right_sensor_value, left_sensor_value)))
        theta += turn_right_small(min(right_sensor_value, left_sensor_value))
    if 'large' in left_sensor_flag and 'small' in right_sensor_flag:
        #print('LS:', turn_left_small(min(right_sensor_value, left_sensor_value)))
        theta += turn_left_small(min(right_sensor_value, left_sensor_value))
        
    if 'small' in front_sensor_flag and 'large' in right_sensor_flag and 'small' not in right_sensor_flag:
        #print('RL:', turn_right_large(min(front_sensor_value, right_sensor_value)))
        theta += turn_right_large(min(front_sensor_value, right_sensor_value))
    if 'small' in front_sensor_flag and 'large' in left_sensor_flag and 'small' not in left_sensor_flag:
        #print('LL:',turn_left_large(min(front_sensor_value, left_sensor_value)))
        theta += turn_left_large(min(front_sensor_value, left_sensor_value))
    
    if theta > 40 :
        theta = 40
    elif theta < -40:
        theta = -40
    #print('theta :', theta)
    return theta

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
    
def GUI(data):
    f = open('train4D.txt', 'w', encoding = 'UTF-8')
    f6 = open('train6D.txt', 'w', encoding = 'UTF-8')
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
            theta = fuzzy(front_distance, right_distance, left_distance)
            f.write(str(front_distance)+' ')
            f.write(str(right_distance)+' ')
            f.write(str(left_distance)+' ')
            f.write(str(theta)+'\n')
            f6.write(str(car.get_x())+' ')
            f6.write(str(car.get_y())+' ')
            f6.write(str(front_distance)+' ')
            f6.write(str(right_distance)+' ')
            f6.write(str(left_distance)+' ')
            f6.write(str(theta)+'\n')
            all_sprites.update(theta)
        #collision
        hits = pygame.sprite.spritecollide(car, walls, False)
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
    f.close()
    f6.close()
    time.sleep(1)
    pygame.quit()
    quit()

def read_files():
    df = pd.read_table('./case01.txt', delimiter  = ',', header = None, keep_default_na = False)
    case01_list = df.values
    return case01_list

def main():
    data = read_files()
    GUI(data)

if __name__ == '__main__':
    main()