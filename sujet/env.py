#import gym
#from gym import spaces
import numpy as np
class Env():
    hello = "coucou"
    conf = None
    def __init__(self, config):
        self.config = config
        #self.action_space = spaces.Discrete(len(config['tasks']))
        #self.observation_space = spaces.Box(low=0,high=1,shape=(len(config['tasks'])))
        self.n_tasks = len(config['tasks'])
        self.operators = np.zeros(config['parameters']['size']['nb_operators'])
        self.machines= np.zeros(config['parameters']['size']['nb_machines'])
        self.started_time= np.full(config['parameters']['size']['nb_tasks'], 999999999)
        self.timetask= np.zeros(config['parameters']['size']['nb_tasks'])
        self.machinestask = np.zeros(config['parameters']['size']['nb_tasks'])
        self.ntasks = config['tasks']
        self.availability = np.zeros((len(self.ntasks),len(self.ntasks)))
        self.jobs= np.zeros((config['parameters']['size']['nb_jobs'],len(self.ntasks)))
        for job in range(len(config['jobs'])):
            seq = config['jobs'][job]['sequence']
            for i in range(len(seq)):
                self.jobs[job,seq[i]-1] = 1;
                for j in range(i+1,len(seq)):
                    self.availability[seq[j]-1,seq[i]-1] = 1;

        self.tasks = []
        for task in range(self.n_tasks):
            self.timetask[task] = config['tasks'][task]['processing_time']
            for machine in self.config['tasks'][task]['machines']:
                for operator in machine['operators']:
                    self.tasks.append([task, machine['machine'], operator])
        self.tasks = np.array(self.tasks)
        self.time = 0
        return;

    def reset(self):

        self.operators = np.zeros(self.config['parameters']['size']['nb_operators']+1)
        self.machines= np.zeros(self.config['parameters']['size']['nb_machines']+1)
        self.started_time= np.full(self.config['parameters']['size']['nb_tasks'],99999999999)
        self.donetasks = np.zeros(self.config['parameters']['size']['nb_tasks'])
        self.donejobs = np.zeros(self.config['parameters']['size']['nb_jobs'])
        self.opmach= np.zeros((self.config['parameters']['size']['nb_tasks'],2),dtype = int)
        print(self.tasks)
        self.time = 0
        self.update_available_mtasks()
        return self.available_mtasks;


    def update_available_mtasks(self):
        ndonetasks = (self.time-self.started_time)>=self.timetask
        recentlydones = np.logical_and(ndonetasks,np.logical_not(self.donetasks))

        for i in np.where(recentlydones)[0]:
            print('done task',i,self.opmach[i])
            self.machines[self.opmach[i][0]]=0
            self.operators[self.opmach[i][1]]=0
            donejobs = np.dot(self.jobs,np.logical_not(ndonetasks).reshape((-1,1)))
            print(self.jobs)
            print(ndonetasks)
            print(donejobs)

        self.donetasks = ndonetasks
        self.available_tasks = np.dot(self.availability,np.logical_not(self.donetasks.reshape((-1,1))))==0
        self.available_mtasks = np.zeros(len(self.tasks))
        #print(self.donetasks,self.available_tasks)
        for i in range(self.tasks.shape[0]):
            task = self.tasks[i]
            if not self.available_tasks[task[0],0]:
                continue
            if self.operators[task[2]]:
                continue
            if self.machines[task[1]]:
                continue
            if self.started_time[task[0]]<=self.time:
                continue
            self.available_mtasks[i]=1

    def step(self, action):
        print('action', action)
        if(action==-1):
            self.time+=1
        else:
            task = self.tasks[action]
            print('task',task)
            print('completing_time',self.timetask[task[0]])
            self.started_time[task[0]] = self.time;
            self.opmach[task[0],0] = task[1];
            self.opmach[task[0],1] = task[2];
            self.operators[task[2]] = 1;
            self.machines[task[1]] = 1;
        self.update_available_mtasks()
        print('tasks updated')
        return self.available_mtasks;
