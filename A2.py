import random
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
        r,c=self.rows-pos[1]-1,pos[0]
        return self.grid[r,c]!=1
    def typeOfCell(self,pos):
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

def reward(s):
    """
    Returns the reward
    s:state given by position tuple(x,y)
    """
    # print(g.typeOfCell(s))
    if g.typeOfCell(s)=="Wall":
        return -1
    elif g.typeOfCell(s)=="Goal":
        return 100
    else:
        return 0
print(reward((48,12)))
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
        if self.grid.typeOfCell(new_pos)!="Wall":
            self.pos=new_pos
        
        return self.pos

a=Agent((1,1),g)
a.take_step('S')
print(a.pos)




# if __name__ == '__main__':
    
