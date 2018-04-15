# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 16:56:25 2018

@author: pig84
"""

import pygame
import pandas as pd
import math

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
    #print('fi : ', output, math.degrees(math.asin(2*math.sin(math.radians(theta))/6)))
    return output

def x_next(x, fi, theta):
    output = x + math.cos(math.radians(fi + theta)) + math.sin(math.radians(theta)) * math.sin(math.radians(fi))
    #print('x : ', output)
    return output

def y_next(y, fi, theta):
    output = y + math.sin(math.radians(fi + theta)) - math.sin(math.radians(theta)) * math.cos(math.radians(fi))
    #print('y : ',output)
    print('-----------')
    return output

def message_display(theta):
    myfont = pygame.font.SysFont('None', 30)
    textsurface = myfont.render('Press Enter to next step. Wheel Degree : '+theta, True, BLACK)
    textrect = textsurface.get_rect()
    textrect.center = (220, 50)
    gameDisplay.blit(textsurface, textrect)
    
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
            denominator +=1
            if j <= 20:
                molecule += ((1/10)*j - 1)*j
            else:
                molecule += j
        return molecule/denominator
    else:
        molecule = 0
        denominator = 0
        for i in range(100, 300):
            j = i/10
            denominator +=1
            if j <= (value*10 - 1):
                molecule += ((1/10)*j - 1)*j
            elif j > (value*10 - 1) and j <= (value - 3)*(-10):
                molecule += value * j 
            else:
                molecule += value*j
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
            denominator +=1
            if j <= 20:
                molecule += ((1/10)*j - 1)*j
            else:
                molecule += j
        return -(molecule/denominator)
    else:
        molecule = 0
        denominator = 0
        for i in range(100, 300):
            j = i/10
            denominator +=1
            if j <= (value*10 - 1):
                molecule += ((1/10)*j - 1)*j
            elif j > (value*10 - 1) and j <= (value - 3)*(-10):
                molecule += value * j 
            else:
                molecule += value*j
        return -(molecule/denominator)
    
def turn_right_large(value):
    if value == 0:
        return 0
    elif value == 1:
        molecule = 0
        denominator = 0
        for i in range(300, 400):
            j = i /10
            denominator +=1
            molecule += j
        return molecule/denominator
    else:
        molecule = 0
        denominator = 0
        for i in range(300, 400):
            j = i /10
            denominator +=1
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
            denominator +=1
            molecule += j
        return -(molecule/denominator)
    else:
        molecule = 0
        denominator = 0
        for i in range(300, 400):
            j = i /10
            denominator +=1
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
    if right_sensor <= 8:
        right_sensor_flag.append('small')
        right_sensor_value = 1
    elif right_sensor > 8 and right_sensor <= 10:
        right_sensor_flag.append('small')
        right_sensor_value = -(1/2) * right_sensor + 5
    #Large
    if right_sensor > 10 and right_sensor <= 12:
        right_sensor_flag.append('large')
        right_sensor_value = -(1/2) * right_sensor + 5
    elif right_sensor > 12:
        right_sensor_flag.append('large')
        right_sensor_value = 1
    ################################################################
    #left
    #Small
    if left_sensor <= 8:
        left_sensor_flag.append('small')
        left_sensor_value = 1
    elif left_sensor > 8 and left_sensor <= 10:
        left_sensor_flag.append('small')
        left_sensor_value = -(1/2) * left_sensor + 5
    #Large
    if left_sensor > 10 and left_sensor <= 12:
        left_sensor_flag.append('large')
        left_sensor_value = -(1/2) * left_sensor + 5
    elif left_sensor > 12:
        left_sensor_flag.append('large')
        left_sensor_value = 1
        
    print('F', front_sensor_flag)
    print('R', right_sensor_flag)
    print('L', left_sensor_flag)
    
    #conclusion
    '''
    if 'large' in front_sensor_flag and 'large' in right_sensor_flag and 'small' not in right_sensor_flag:
        print(turn_right_small(min(front_sensor_value, right_sensor_value)))
        theta += turn_right_small(min(front_sensor_value, right_sensor_value))
    if 'large' in front_sensor_flag and 'large' in left_sensor_flag and 'small' not in left_sensor_flag:
        print(turn_left_small(min(front_sensor_value, left_sensor_value)))
        theta += turn_left_small(min(front_sensor_value, left_sensor_value))
    '''
    if 'large' in right_sensor_flag and 'small' in left_sensor_flag:
        print('RS:', turn_right_small(min(right_sensor_value, left_sensor_value)))
        theta += turn_right_small(min(right_sensor_value, left_sensor_value))
    if 'large' in left_sensor_flag and 'small' in right_sensor_flag:
        print('LS:', turn_left_small(min(right_sensor_value, left_sensor_value)))
        theta += turn_left_small(min(right_sensor_value, left_sensor_value))
        
    if 'small' in front_sensor_flag and 'large' in right_sensor_flag and 'small' not in right_sensor_flag:
        print('RL:', turn_right_large(min(front_sensor_value, right_sensor_value)))
        theta += turn_right_large(min(front_sensor_value, right_sensor_value))
    if 'small' in front_sensor_flag and 'large' in left_sensor_flag and 'small' not in left_sensor_flag:
        print('LL:',turn_left_large(min(front_sensor_value, left_sensor_value)))
        theta += turn_left_large(min(front_sensor_value, left_sensor_value))
    
    if theta > 40 :
        theta = 40
    elif theta < -40:
        theta = -40
    print('theta :', theta)
    return theta

class Car(pygame.sprite.Sprite):
    def __init__(self, init_value):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((6*SCALE, 6*SCALE))
        #self.image.fill(WHITE)
        #self.image = pygame.Surface((50, 50), pygame.SRCALPHA, 32)
        #self.image = self.image.convert_alpha()
        self.rect = self.image.get_rect()
        self.pos_x = int(init_value[0])
        self.pos_y = int(init_value[1])
        self.fi = int(init_value[2])
        self.rect.center = (self.pos_x + X_ZERO, self.pos_y + Y_ZERO)
        pygame.draw.circle(self.image, RED, (int(3*SCALE), int(3*SCALE)), 3*SCALE, 3) 
        pygame.draw.line(self.image, RED, (int(3*SCALE), int(3*SCALE)),
                        (3*SCALE*math.cos(math.radians(self.fi)) + int(3*SCALE), -3*SCALE*math.sin(math.radians(self.fi))+ int(3*SCALE)), 3) 
    
    def update(self, theta):
        #render update
        self.image.fill(BLACK)
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
            self.image = pygame.Surface((1*SCALE, abs(start_value[1] - stop_value[1])*SCALE))
            self.rect = self.image.get_rect()
            x_mean = (start_value[0] + stop_value[0])/2*SCALE
            y_mean = (start_value[1] + stop_value[1])/2*SCALE
            self.rect.center = (x_mean + X_ZERO, -y_mean + Y_ZERO)
            
            #pygame.draw.line(self.image, RED, (start_value[0]*SCALE, start_value[1]*SCALE), (stop_value[0]*SCALE, stop_value[1]*SCALE), 3)
        #horizon
        elif (start_value[1] == stop_value[1]):
            self.image = pygame.Surface(((abs(start_value[0] - stop_value[0]) + 1)*SCALE, 1*SCALE))
            self.rect = self.image.get_rect()
            x_mean = (start_value[0] + stop_value[0])/2*SCALE
            y_mean = (start_value[1] + stop_value[1])/2*SCALE
            self.rect.center = (x_mean + X_ZERO, -y_mean + Y_ZERO)
            
            #pygame.draw.line(self.image, RED, (start_value[0]*SCALE, start_value[1]*SCALE), (stop_value[0]*SCALE, stop_value[1]*SCALE), 3)
        else:
            self.image = pygame.Surface((abs(start_value[0] - stop_value[0])*SCALE, 1*SCALE))
            #self.image.fill(WHITE)
            self.rect = self.image.get_rect()
            x_mean = (start_value[0] + stop_value[0])/2*SCALE
            y_mean = ((start_value[1] + stop_value[1])/2)*SCALE
            self.rect.center = (x_mean + X_ZERO, -y_mean + Y_ZERO)
            pygame.draw.line(self.image, RED, (start_value[0]*SCALE, start_value[1]*SCALE), (stop_value[0]*SCALE, stop_value[1]*SCALE), 3)

    def get_points(self):
        return (self.point_1, self.point_2)
    
def GUI(data):
    f = open('B.txt', 'w', encoding = 'UTF-8')
    n, d = data.shape
    theta = 0
    #sprites
    all_sprites = pygame.sprite.Group()
    walls = pygame.sprite.Group()
    car = Car(data[0])
    all_sprites.add(car)
    #wall = Wall(data[3], data[4])
    for i in range(3, n-1):
        #print(data[i], data[i+1])
        wall = Wall(data[i], data[i+1])
        all_sprites.add(wall)
        walls.add(wall)
    
    #game loop
    running = True
    while running:
        #FPS
        clock.tick(60)
        
        #get sensor
        front_distance = car.sensor(walls.sprites(), 'front')
        right_distance = car.sensor(walls.sprites(), 'right')
        left_distance = car.sensor(walls.sprites(), 'left')
        
        #process input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    theta += 5
                    if theta > 40:
                        theta = 40
                elif event.key == pygame.K_LEFT:
                    theta -= 5     
                    if theta < -40:
                        theta = -40
                elif event.key == pygame.K_RETURN:
                    all_sprites.update(fuzzy(front_distance, right_distance, left_distance))
                    #fuzzy(front_distance, right_distance, left_distance)
                    print(front_distance, right_distance, left_distance, car.get_fi())
                    f.write(str(front_distance)+' ')
                    f.write(str(right_distance)+' ')
                    f.write(str(left_distance)+' ')
                    f.write(str(car.get_fi())+'\n')
                    theta = 0
        #update
        
        #collision
        hits = pygame.sprite.spritecollide(car, walls, False)
        if hits:
            running = False
        
        #render
        gameDisplay.fill(WHITE)
        message_display(str(theta))
        all_sprites.draw(gameDisplay)
        pygame.display.update()
    f.close()
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