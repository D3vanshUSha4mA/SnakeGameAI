import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI,Direction,Point
from model import Linear_QNet,QTrainer
from helper import plot     
MAX_MEMORY=100_000
BATCH_SIZE=1000
LR=0.001

class Agent:
    def __init__(self):
        self.n_games=0
        self.epsilon=0
        self.gamma=0.9
        self.memory=deque(maxlen=MAX_MEMORY)
        self.model=Linear_QNet(11,256,3)     #NN input is 11(state) and output is 3(staright,right,left)
        self.trainer=QTrainer(self.model,lr=LR,gamma=self.gamma)
        

    def get_state(self,game):
        head=game.snake[0]
        point_l=Point(head.x-20,head.y)
        point_r=Point(head.x+20,head.y)
        point_u=Point(head.x,head.y-20)
        point_d=Point(head.x,head.y+20)
        dir_l=game.direction==Direction.LEFT
        dir_r=game.direction==Direction.RIGHT
        dir_u=game.direction==Direction.UP
        dir_d=game.direction==Direction.DOWN
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x<game.head.x,  # food left
            game.food.x>game.head.x,  # food right
            game.food.y<game.head.y,  # food up
            game.food.y>game.head.y  # food down
            ]

        return np.array(state, dtype=int)
    

    def remember(self,state,action,reward,next_state,done):                #stores a single experience
        self.memory.append((state,action,reward,next_state,done))
    
    def train_long_memory(self):                                           #trains on a batch of old experiences from the memory                                    
        if len(self.memory)>BATCH_SIZE:
            mini_sample=random.sample(self.memory,BATCH_SIZE)
        else:
            mini_sample=self.memory
        states,actions,rewards,next_states,dones=zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states,dones)
 
    def train_short_memory(self,state,action,reward,next_state,done):        #trains the model on this one experience 
        self.trainer.train_step(state,action,reward,next_state,done)
    def get_action(self,state):
        self.epsilon=80-self.n_games
        final_move=[0,0,0]
        if random.randint(0,200)<self.epsilon:                              #random no from 0 to 200 is drawn if it's less than epsilon then exploration is done..
            move=random.randint(0,2)
            final_move[move]=1
        else:                                                               #choses best action here
            state0=torch.tensor(state,dtype=torch.float)                    #unput state into a pytorch tensor of float type
            prediction=self.model(state0)                                   #predict q-values for the three actions
            move=torch.argmax(prediction).item()                            #chooses with the one having maximum q-value(index)
            final_move[move]=1                                              #updates into our fnal_move vector  
        return final_move

def train():
    plot_scores=[]                                                  #scores of each game  
    plot_mean_scores=[]                                             #average score over time
    total_score=0                                                   #running total of all game scores
    record=0                                                        #highest score till now
    agent=Agent()                                                   #DQN Agent
    game=SnakeGameAI()                                              
    while True:
        state_old=agent.get_state(game)                             #current game situation
        final_move=agent.get_action(state_old)                      #pick an action
        reward,done,score=game.play_step(final_move)                #takes an action in that state
        state_new=agent.get_state(game)                             #new state after action 
        agent.train_short_memory(state_old,final_move,reward,state_new,done)  #immediately learns from this experience 
        agent.remember(state_old,final_move,reward,state_new,done)            #experience stored in replay memory
        if done:                                   #if snake is dead
            game.reset()                            #restart a new game..
            agent.n_games+=1
            agent.train_long_memory()               #samples a batch of experiences from memory buffer and trains model on all of them

            if score>record:                        #save best model....
                record=score
                agent.model.save()
            print('Game',agent.n_games,'Score',score,'Record:',record)         #track and plot progress
            plot_scores.append(score)
            total_score+=score
            mean_score=total_score/agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores)
            


if __name__=='__main__':
    train()
