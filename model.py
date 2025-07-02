import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.linear1=nn.Linear(input_size,hidden_size)   #input to hidden layer
        self.linear2=nn.Linear(hidden_size,output_size)  #hidden layer to output

    def forward(self,x):
        x=F.relu(self.linear1(x))    #input value is passed through first layer and then activation function is applied which converts neg values to zero and adds non-linearity..
        x=self.linear2(x)            #passed through second layer... 
        return x                     #calculated Q-value is returned..
    
    def save(self,file_name='model.pth'):  #saves the model
        model_folder_path='./model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name=os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(),file_name)
    
class QTrainer:
    def __init__(self,model,lr,gamma):     
        self.lr=lr
        self.gamma=gamma
        self.model=model
        self.optimizer=optim.Adam(model.parameters(),lr=self.lr)   #Adam intializes the optimizer which update model parameters during training....
        self.criterion=nn.MSELoss()                                #mean squared error loss function
    
    def train_step(self,state,action,reward,next_state,done):
        state=torch.tensor(state,dtype=torch.float)                #converts inputs to tensors
        next_state=torch.tensor(next_state,dtype=torch.float)
        action=torch.tensor(action,dtype=torch.long)
        reward=torch.tensor(reward,dtype=torch.float)

        if len(state.shape)==1:                                    #if state is a single sample then it is converted to 2-D tensor using unsqueeze..
            state=torch.unsqueeze(state,0)
            next_state=torch.unsqueeze(next_state,0)
            action=torch.unsqueeze(action,0)
            reward=torch.unsqueeze(reward,0)
            done=(done,)

        pred=self.model(state)    #prediction is taken of current state by model forward pass.....returns a tensor containing Q-values of all actions
        target=pred.clone()       #a clone is made 
        for idx in range(len(done)):
            Q_new=reward[idx]       #if game over q-value if reward
            if not done[idx]:       #if game not over then estimate future reward
                Q_new=reward[idx]+self.gamma*torch.max(self.model(next_state[idx]))     #bellman eqn(torch.max->maximum q-values among all possible action in next state)
            action_idx=torch.argmax(action[idx]).item()  #returns the index of 1(action taken)
            target[idx][action_idx]=Q_new                #only change value for those which are actually taken


        self.optimizer.zero_grad()                       #clear gradients of last iterations..
        loss=self.criterion(pred,target)                 
        loss.backward()
        self.optimizer.step()



