import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI:
    
    def __init__(self,w=640,h=480):
        self.w=w
        self.h=h
        # init display
        self.display=pygame.display.set_mode((self.w,self.h))
        pygame.display.set_caption('Snake')
        self.clock=pygame.time.Clock()
        self.reset()    #to start the game fresh
    
    def reset(self):
        # init game state
        self.direction=Direction.RIGHT  #initial direction of snake
        
        self.head=Point(self.w/2, self.h/2)  #head in the middle
        self.snake=[self.head,               #snake...
                      Point(self.head.x-BLOCK_SIZE,self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE),self.head.y)]
        
        self.score=0
        self.food=None
        self._place_food()
        self.frame_iteration=0          #count no of steps

    def _place_food(self):              #random food placement on the grid
        x=random.randint(0,(self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE 
        y=random.randint(0,(self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food=Point(x, y)
        if self.food in self.snake:
            self._place_food()          #recursive call if point is on the snake
        
    def play_step(self,action):         #takes an action and returns a reward,game_over and score.... 
        self.frame_iteration+=1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
           
        
        self._move(action)                    #update the head
        self.snake.insert(0, self.head)       #adds new head position at front index of the snake list
        
        
        reward=0
        game_over=False
        if self.is_collision() or self.frame_iteration>100*len(self.snake):  #if there is a collision or too many loops without eating...
            game_over=True
            reward=-10
            return reward,game_over,self.score
            
        
        if self.head==self.food:          #if food is eaten then place new food
            self.score+=1
            reward=10
            self._place_food()
        else:
            self.snake.pop()              #else removes tail to simulate forward movement
        
        
        self._update_ui()
        self.clock.tick(SPEED)
        return reward,game_over,self.score
    
    def is_collision(self,pt=None):
        if pt is None:
            pt=self.head
        if pt.x>self.w-BLOCK_SIZE or pt.x<0 or pt.y>self.h-BLOCK_SIZE or pt.y<0:   #out of bounds
            return True
        if pt in self.snake[1:]:          #if snake hits it's itself
            return True
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display,BLUE1,pygame.Rect(pt.x,pt.y,BLOCK_SIZE,BLOCK_SIZE))
            pygame.draw.rect(self.display,BLUE2,pygame.Rect(pt.x+4,pt.y+4,12,12))
            
        pygame.draw.rect(self.display,RED,pygame.Rect(self.food.x,self.food.y,BLOCK_SIZE,BLOCK_SIZE))
        
        text=font.render("Score: "+str(self.score),True,WHITE)
        self.display.blit(text,[0,0])
        pygame.display.flip()
        
    def _move(self,action):
        clock_wise=[Direction.RIGHT,Direction.DOWN,Direction.LEFT,Direction.UP]
        idx=clock_wise.index(self.direction)
        if np.array_equal(action,[1,0,0]):         #go straight,direction does not change.....
            new_dir=clock_wise[idx]
        if np.array_equal(action,[0,1,0]):         #turn right...
            next_idx=(idx+1)%4                     #one step clockwise.....
            new_dir=clock_wise[next_idx]
        if np.array_equal(action,[0,0,1]):         #turn left...
            next_idx=(idx-1)%4                     #one step anticlockwise..
            new_dir=clock_wise[next_idx]
        
        self.direction=new_dir                      #update direction..
        x=self.head.x
        y=self.head.y
        if self.direction==Direction.RIGHT:         #update the position of head....
            x+=BLOCK_SIZE
        elif self.direction==Direction.LEFT:
            x-=BLOCK_SIZE
        elif self.direction==Direction.DOWN:
            y+=BLOCK_SIZE
        elif self.direction==Direction.UP:
            y-=BLOCK_SIZE
            
        self.head=Point(x, y)
            

if __name__ == '__main__':
    game = SnakeGameAI()
    
    # game loop
    while True:
        game_over,score=game.play_step()
        
        if game_over==True:
            break
        
    print('Final Score', score)
        
        
    pygame.quit()