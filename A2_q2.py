import random
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

random.seed(0)
actions={'E':(1,0),'N':(0,1),'W':(-1,0),'S':(0,-1)}
action_int_to_str = {0:'E',1:'N',2:'S',3:'W'}
action_str_to_int = {'E':0,'N':1,'S':2,'W':3}

print("Testing Begins")

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

# print(w1)
g=Grid(50,25,w1,goal=(48,12))
print(g)
g.show()

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

print(transition((1,1),'N',(2,1)))

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
        new_pos=self.pos[0]+actions[act][0],self.pos[1]+actions[act][1]
        
        status = "Wall"
        if self.grid.typeOfCell(new_pos)!="Wall":
            self.pos=new_pos
            status = "Fine"
            
        
        return self.pos,status

a=Agent((1,1),g)
pos,status = a.take_step('W')
print(pos,status)
print("Testing Ends")

print("Part-2 Q Learning Begins")

def xytoRC(pos,rows,cols):
    """ XY to RC Transformation """
    row,col=rows-pos[1]-1,pos[0]
    return row,col

def action_extractor(Q,row,col,epsilon):
    """
    Parameters
    ----------
    Q : NP matrix
        Q(s,a) matrix.
    row : int
        row no. of state under observation.
    col : int
        col no. of state under observation
    epsilon : float
        Wieght to give to choosing the best action.

    Returns
    -------
    string
        Action to be taken.

    """
    
    best_q = -1000000000
    best_action = -1
    for var in range(len(actions.keys())):
        if Q[row][col][var]>best_q :
            best_q = Q[row][col][var]
            best_action = var
            
    l = [0,1]
    wts = [1-epsilon,epsilon]
    
    num_choosed=random.choices(l,wts)[0]
    
    if num_choosed == 0:
        return action_int_to_str[best_action]
    else:
        rand_act= random.choice('NSEW')
        return rand_act
    
def best_q_val_extractor(Q,row,col):
    """
    Parameters
    ----------
    Q : NP matrix
        Q(s,a) matrix.
    row : int
        row no. of state under observation.
    col : int
        col no. of state under observation
    
    Returns
    -------
    float
        Q val corresponding to best action.

    """
    
    best_q = -1000000000
    for var in range(len(actions.keys())):
        if Q[row][col][var]>best_q :
            best_q = Q[row][col][var]
               
    return best_q

def best_q_action_extractor(Q,row,col):
    """
    Parameters
    ----------
    Q : NP matrix
        Q(s,a) matrix.
    row : int
        row no. of state under observation.
    col : int
        col no. of state under observation
    
    Returns
    -------
    float
        Q val corresponding to best action.
        
    str 
        best action
    """
    
    best_q = -1000000000
    best_action = -1
    for var in range(len(actions.keys())):
        if Q[row][col][var]>best_q :
            best_q = Q[row][col][var]
            best_action = var
            
   
    return best_q,best_action
    
def q_learning(num_episodes,max_steps,alpha,epsilon,gamma,agent):
    """
    Parameters
    ----------
    num_episodes : int
        Number of episodes for algo.
    max_steps : int
        Maximum Number of steps in an episode
    alpha : float
        learning rate for Q-Learning
    epsilon : float
        Exploration constant for Q-Learning
    gamma : float
        Discount factor
    agent : Class agent
        The agent which is traversing the grid.
    
    
    Returns
    -------
    Optimal Policy, Value Function.

    """
    
    rows = agent.grid.rows
    cols = agent.grid.cols
    Q = np.random.randn(rows,cols,len(actions.keys()))
    V = np.random.randn(rows,cols)
    
    goal_x,goal_y = agent.grid.goal
    row_g,col_g = xytoRC((goal_x,goal_y),rows,cols)
    for var in range(len(actions.keys())):
        Q[row_g][col_g][var]=0
        
    
    policy = [['-' for i in range(cols)] for j in range(rows)]
    
    
    for episode_no in range(num_episodes):
        
        x,y=random.randint(1,48),random.randint(1,23)
        pos=(x,y)
        while agent.grid.typeOfCell(pos)=='Wall':
            x-=3
            pos=(x,y)
        row,col = xytoRC((x,y),rows,cols)
        agent.pos = pos
        
        for step in range(max_steps):
            
            action=action_extractor(Q, row, col, epsilon)   
            pos_next,status = agent.take_step(action)
            row_next,col_next = xytoRC(pos_next,rows,cols)
            
            reward = 0
            if status == "Wall":
                reward = -1  
            if pos_next == agent.grid.goal :
                reward = 100
            
            Q[row][col][action_str_to_int[action]] = Q[row][col][action_str_to_int[action]] \
                + alpha*reward \
                - alpha*Q[row][col][action_str_to_int[action]] \
                + alpha*gamma*best_q_val_extractor(Q, row_next, col_next)
              
            pos = pos_next
            row = row_next
            col = col_next
            
            if reward == 100:
                break
    
    for x_v in range(cols):
        for y_v in range(rows):
            row_v,col_v = xytoRC((x_v,y_v), rows, cols)
            V[row_v][col_v], p_ret= best_q_action_extractor(Q, row_v, col_v)
            
            if agent.grid.typeOfCell((x_v,y_v)) != "Wall":
                policy[row_v][col_v] = action_int_to_str[p_ret]
    
    return policy,V
            

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
    
    
policy,V= q_learning(4000,1000,0.25, 0.5, 0.99, a)

min_val = np.amin(V)
V = V - min_val
plt.imshow(V,cmap='gray')
plt.show()

policyPlot(policy, 25, 50)