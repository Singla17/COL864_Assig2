import random
import stat
from typing import Tuple
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sympy import re

random.seed(0)
actions={'E':(1,0),'N':(0,1),'W':(-1,0),'S':(0,-1)}

class Grid:
    def __init__(self,xdim,ydim,walls=[],add_boundary=True,goal=None):
        self.rows = ydim
        self.cols=xdim
        self.grid=np.zeros((self.rows,self.cols))
        self.walls=(walls)
        if add_boundary:
            for i in range(self.rows):
                self.grid[i,0]=1
                self.grid[i,-1]=1
            for i in range(self.cols):
                self.grid[0,i]=1
                self.grid[-1,i]=1
        if goal:
            self.goal=goal
            goal_x=goal[0]
            goal_y=goal[1]
            
            goal_r=self.rows-goal_y-1
            goal_c=goal_x
            self.grid[goal_r,goal_c]=-1
        for wall in walls:
            x=wall[0]
            y=wall[1]
            r=self.rows-y-1
            c=x
            # print(r,c)
            if r>=0 and c>=0:
                self.grid[r,c]=1
    
    def isValid(self,pos):
        """ XY Input """
        r,c=self.rows-pos[1]-1,pos[0]
        if r >=0 and r < self.rows and c>=0 and c<self.cols:
            return True
        return False
    
    def typeOfCell(self,pos):
        """ XY Input """
        
        assert self.isValid(pos), "Cell Number Out of Bounds"
        r,c=self.rows-pos[1]-1,pos[0]
        if self.grid[r,c]==-1:
            return "Goal"
        elif self.grid[r,c]==1:
            return "Wall"
        else :
            return 'Empty'
        
        
    def show(self):
        fig, ax = plt.subplots()
        plt.xlim(-0.5,self.cols-0.5)
        plt.ylim(-0.5,self.rows-0.5)
        # plt.axis(False)
        
        for r in range (self.rows):
            for c in range(self.cols):
                x,y=c,self.rows-r-1
                if self.grid[r,c]==1:
                    ax.add_patch(Rectangle((x-0.5,y-0.5),
                            1, 1,
                            fc ='grey', 
                            lw = None))
                elif self.grid[r,c]==0:
                    ax.add_patch(Rectangle((x-0.5,y-0.5),
                            1, 1,
                            fc ='none',
                            lw = None))
                else:
                    ax.add_patch(Rectangle((x-0.5,y-0.5),
                            1, 1,
                            fc ='red',
                            lw = None))
        
        xticks=np.arange(0,self.cols,5)
        yticks=np.arange(0,self.rows,5)
        plt.xticks(xticks)
        plt.yticks(yticks)
        # plt.grid(True)
        plt.show()
    def __str__(self):
        return str(self.grid)

w1=[(25,i) for i in range(1,12)]
w2=[(25,i) for i in range(13,25)]
w1.extend(w2)
w3=[(26,i) for i in range(1,12)]
w4=[(26,i) for i in range(13,25)]
w3.extend(w4)
w1.extend(w3)

g=Grid(50,25,w1,goal=(48,12))
# g.show()

def transition(s,a,s_dash):
    """
    s:initial state given by position tuple(x,y)
    a:Action is a letter from [N,S,E,W]
    s_dash:final state given by position tuple(x,y)
    """
    x,y=s[0],s[1]
    x_n,y_n=s_dash[0],s_dash[1]
    if (x_n-x,y_n-y) not in list(actions.values()):
        return 0
    else:
        if x+actions[a][0]==x_n and y+actions[a][1]==y_n:
            return 0.8
        else:
            return 0.2/3

# print(transition((1,1),'E',(2,1)))

def reward(g,s):
    """
    Returns the reward
    s:state given by position tuple(x,y)
    """
    if g.typeOfCell(s)=="Wall":
        return -1
    elif g.typeOfCell(s)=="Goal":
        return 100
    else:
        return 0
print(reward(g,(48,12)))

class Agent:
    def __init__(self,init_pos,grid,path=[]) :
        self.pos = init_pos
        self.path = path
        self.grid=grid

    def take_step(self,a):
        """
        Takes a step for the agent
        a:Action given by letter {N,S,E,W}
        """
        l=list(actions.keys())
        wts=[]
        for action in l:
            prob=0.8 if action==a else 0.2/3
            wts.append(prob)
        act=random.choices(l,wts)[0]
        prob=0.8 if act==a else 0.2/3
        # print(a,act,prob)
        new_pos=self.pos[0]+actions[act][0],self.pos[1]+actions[act][1]
        if self.grid.typeOfCell(new_pos)!="Wall":
            self.pos=new_pos
        
        return self.pos,new_pos

# a=Agent((1,1),g)
# a.take_step('N')
# print(a.pos)

# point = (25,13)
# print(a.grid.isValid(point))
# print(a.grid.typeOfCell(point))

print("Testing Ends")

def xytoRC(pos,rows,cols):
    """ XY to RC Transformation """
    row,col=rows-pos[1]-1,pos[0]
    return row,col

def balanced_wandering(pos:tuple,g,counts:dict)->str:
    """Given a state ,the balanced wandering policy selects an action out of {N,S,E,W}
    based on if it has been visited or not 

    Args:
        pos (tuple): state (x,y)
        counts (dict): count[state][N]: tells number of North actions taken at this state
    """
    assert g.typeOfCell(pos) not in ['Wall','Goal'],f"Can't wander from wall or terminal goal state : {pos} "
    if pos in counts:
        min_act,min_count='X',1e9
        for act in "ENSW":
            cnt=counts[pos].get(act,0)
            if cnt<min_count:
                min_count=cnt
                min_act=act
        counts[pos][min_act]=counts[pos].get(min_act,0)+1
        return min_act
    else:
        act= random.choice('NSEW')
        counts[pos]={}
        counts[pos][act]=1
        return act

# counts={(1, 1): {'W': 1}}

# print(balanced_wandering((1,1),g,counts))
# print(counts)

state_count={}
for x in range(0,50):
    for y in range(0,25):
        pos=(x,y)
        if g.typeOfCell(pos)!='Wall':
            state_count[pos]=0
# print(state_count.keys())
# num_eps=1
# max_sim_len=30
num_eps=100
max_sim_len=1000
stds=[]
mean_visits_per_state=[]
episodes=[]
state_count={}

for _ in range(num_eps):
    episode=[]
    x,y=random.randint(1,48),random.randint(1,23)
    pos=(x,y)
    # x,y=48,13
    # pos=(x,y)
    while g.typeOfCell(pos)=='Wall':
        x-=3
        pos=(x,y)
    
    counts={}
    a=Agent(pos,g)

    for _ in range(max_sim_len):
        state_count[pos]=state_count.get(pos,0)+1
        episode.append(pos)
        state_count[pos]+=1
        if g.typeOfCell(pos)=="Goal":
            break
        action=balanced_wandering(pos,g,counts)
        episode.append(action)
        final_pos,poss_state=a.take_step(action)
        episode.append(reward(g,poss_state))
        pos=final_pos
    mean_visits_per_state.append(sum(list(state_count.values()))/len(list(state_count.keys())))
    temp=np.array(list(state_count.values()))
    stds.append(np.std(temp))
    episodes.append(episode)
reward_dict={}
fig,ax=plt.subplots()
x=np.arange(1,num_eps+1)
plt.plot(x,mean_visits_per_state)
plt.xlabel('Episodes')
plt.ylabel('Mean Visits per state')
plt.title('MEAN VISITS PER STATE')
plt.errorbar(x, mean_visits_per_state, yerr = stds, fmt ='o')
plt.show()
def reward_estimate(grid,state:tuple)-> float:
    """Gives estimated reward for state given from episodes

    Args:
        state (tuple): Position(x,y)

    Returns:
        float: reward for state given from episodes
    """    
    if state in reward_dict:
        return reward_dict[state]
    else:
        rew_sum=0
        num_state=0
        for episode in episodes:
            for i in range(3,len(episode),3):
                if episode[i]==state:
                    rew_sum+=episode[i-1]
                    num_state+=1
        reward_dict[state]=rew_sum/num_state if num_state!=0 else 0
        return reward_dict[state]

# print(reward_estimate((48,13)))
# print(episodes[0])
transition_dict={}
def transition_estimate(s:tuple,a:str,s_dash:tuple)->float:
    """Calculates the transition probabilities for the transition (s,a,s_dash)

    Args:
        s (tuple): State
        a (str): Action
        s_dash (tuple):State reached

    Returns:
        float: transition probability
    """
    if (s,a,s_dash) in transition_dict:
        return transition_dict[(s,a,s_dash)]
    else:
        num_s_dash_visits=0
        num_state_action=0
        for episode in episodes:
            for i in range(0,len(episode)-3,3):
                if (episode[i],episode[i+1]) == (s,a):
                    num_state_action+=1
                    if episode[i+3]==s_dash:
                        num_s_dash_visits+=1
        
        transition_dict[(s,a,s_dash)]=num_s_dash_visits/num_state_action if num_state_action !=0 else 0
        return transition_dict[(s,a,s_dash)]
# print(transition_estimate((48,18),'E',(47,18)))
def value_iteration(thresh,agent,gamma,num_itrs,transition,reward):
    """
    Parameters
    ----------
    thresh : float
        thershold to decide conitnuation of algo.
    agent : Class agent
        The agent which is traversing the grid.
    gamma : float
        discount value.
    num_itrs : int
        maximum number of iterations to be allowed.

    Returns
    -------
    Optimal Policy, Value Function.

    """
    
    rows = agent.grid.rows
    cols = agent.grid.cols
    V_init = np.random.randn(rows,cols)
    V_k = V_init
    policy = [['-' for i in range(cols)] for j in range(rows)]
    
    itr = 0
    delta = thresh + 1 ## Just to get past the while condition on first epoch

    while delta > thresh and itr < num_itrs:
        # print(itr)
        delta = 0
        V_k_1 = np.zeros((rows,cols))
        
        for x in range(cols):
            for y in range(rows):
                
                row,col = xytoRC((x,y),rows,cols)
                
                v = V_k[row][col]
                s_bar_arr = []
                
                x,y = col,rows-row-1
                
                if agent.grid.typeOfCell((x,y)) == "Wall":
                    V_k_1[row][col] = -1
            
                else:
                    if agent.grid.isValid((x+1,y)):
                        s_bar_arr.append((x+1,y))
                        
                    if agent.grid.isValid((x,y+1)):
                        s_bar_arr.append((x,y+1))
                    
                    if agent.grid.isValid((x-1,y)):
                        s_bar_arr.append((x-1,y))
                        
                    if agent.grid.isValid((x,y-1)):
                        s_bar_arr.append((x,y-1))
                    
            
                    temp_var = -10000
                    best_action = "N"
                    for action in actions :
                        
                        sum_var = 0
                        for s_bar in s_bar_arr:
                            
                            if agent.grid.typeOfCell(s_bar) == "Wall":
                               sum_var += transition((x,y),action,s_bar)*(reward(agent.grid,s_bar)+gamma*V_k[row][col]) 
                            else:
                                r_s_bar,c_s_bar = xytoRC(s_bar,rows,cols)
                                sum_var += transition((x,y),action,s_bar)*(reward(agent.grid,s_bar)+gamma*V_k[r_s_bar][c_s_bar])
                        temp_var_prev = temp_var
                        temp_var = max(temp_var,sum_var)
                        
                        if temp_var_prev != temp_var:
                            best_action = action
                    
                    V_k_1[row][col]= temp_var
                    policy[row][col]= best_action
                    delta = max(delta,abs(v-temp_var))
                
        V_k = V_k_1.copy()
        
        V_k_plot = V_k.copy()
        min_val = np.amin(V_k_plot)
        V_k_plot = V_k_plot - min_val
        plt.imshow(V_k_plot,cmap='gray')
        plt.show()   
        itr += 1
        
    print("The total Number of iterations taken were: "+str(itr))
    return policy,V_k

# policy,V= value_iteration(0.1, a, 0.99, 100,transition_estimate,reward_estimate)


def policyPlot(policy,rows,cols):
    
   
    fig, ax = plt.subplots()
    plt.xlim(-0.5,cols-0.5)
    plt.ylim(-0.5,rows-0.5)
    # plt.axis(False)
    
    for r in range (rows):
        for c in range(cols):
            x,y=c,rows-r-1
            if policy[r][c]=='-':
                ax.add_patch(Rectangle((x,y),
                        1, 1,
                        fc ='grey', 
                        lw = None))
            
            elif policy[r][c]=='N':
                ax.arrow(x+0.4, y+0.15,0,0.7,
                        width = 0.2, length_includes_head=True, color ='red')
                
            elif policy[r][c]=='S':
                ax.arrow(x+0.4, y+0.85,0,-0.7,
                        width = 0.2, length_includes_head=True, color ='olive')
            elif policy[r][c]=='E':
                ax.arrow(x+0.15, y+0.4,0.7,0,
                        width = 0.2, length_includes_head=True, color ='blue')
            elif policy[r][c]=='W':
                ax.arrow(x+0.85, y+0.4,-0.70,0,
                        width = 0.2, length_includes_head=True, color ='black')
            else:
                ax.add_patch(Rectangle((x,y),
                        1, 1,
                        fc ='none',
                        lw = None))
    
    major_xticks=np.arange(0,cols,5)
    minor_xticks = np.arange(0,cols,1)
    minor_yticks=np.arange(0,rows,1)
    major_yticks=np.arange(0,rows,5)
    
    ax.set_xticks(major_xticks)
    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks(major_yticks)
    ax.set_yticks(minor_yticks, minor=True)
    
    ax.grid(which='both')
    
    plt.show()
    
    
# policyPlot(policy,25,50)
            
            