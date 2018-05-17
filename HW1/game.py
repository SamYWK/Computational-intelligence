# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 16:15:58 2018

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

def fi_next(fi, theta):
    output = fi - math.degrees(math.asin(2*math.sin(math.radians(theta))/36))
    print('fi : ', output, math.degrees(math.asin(2*math.sin(math.radians(theta))/36)))
    return output

def x_next(x, fi, theta):
    output = x + math.cos(math.radians(fi + theta)) + math.sin(math.radians(theta)) * math.sin(math.radians(fi))
    print('x : ', output)
    return output

def y_next(y, fi, theta):
    output = y + math.sin(math.radians(fi + theta)) - math.sin(math.radians(theta)) * math.cos(math.radians(fi))
    print('y : ',output)
    print('-----------')
    return output

def message_display(theta):
    myfont = pygame.font.SysFont('None', 30)
    textsurface = myfont.render('Press Enter to next step. Wheel Degree : '+theta, True, BLACK)
    textrect = textsurface.get_rect()
    textrect.center = (220, 50)
    gameDisplay.blit(textsurface, textrect)
    

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
    
    def update(self, x, y):
        #render update
        self.image.fill(BLACK)
        pygame.draw.circle(self.image, RED, (int(3*SCALE), int(3*SCALE)), 3*SCALE, 3) 
        pygame.draw.line(self.image, RED, (int(3*SCALE), int(3*SCALE)),
                        (3*SCALE*math.cos(math.radians(self.fi)) + int(3*SCALE), -3*SCALE*math.sin(math.radians(self.fi))+ int(3*SCALE)), 3) 
        
        #parameter update
        #self.fi = fi_next(self.fi, theta)
        self.pos_x += x
        self.pos_y += y
        self.rect.center = (self.pos_x*SCALE + X_ZERO, -self.pos_y*SCALE + Y_ZERO)
        
    def get_x(self):
        return self.pos_x
    
    def get_y(self):
        return self.pos_y
        
            
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

def GUI(data):
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
        #process input
        for event in pygame.event.get():
            x = 0
            y = 0
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    x = 1
                    all_sprites.update(x, y)
                elif event.key == pygame.K_LEFT:
                    x = -1 
                    all_sprites.update(x, y)
                elif event.key == pygame.K_UP:
                    y = 1
                    all_sprites.update(x, y)
                elif event.key == pygame.K_DOWN:
                    y = -1
                    all_sprites.update(x, y)
        #update
        
        
        #collision
        #hits = pygame.sprite.spritecollide(car, walls, False)
        hits = collision(car.get_x(), car.get_y(), walls)
        if hits:
            running = False
        
        #render
        gameDisplay.fill(WHITE)
        message_display(str(theta))
        all_sprites.draw(gameDisplay)
        pygame.display.update()
        
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