use serde::{Deserialize, Serialize};
use std::env;
use std::cmp;
use std::time::{Duration, SystemTime};
use ndarray::prelude::*;
use rand::prelude::*;
use rand::distributions::WeightedIndex;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    dbg!(&args);
    let file_path = &args[1];
    println!("In file {}", file_path);

    let contents = fs::read_to_string(file_path)
        .expect("Should have been able to read the file");
    let v: MyEnv = match serde_json::from_str(&contents){
        Ok(v) => {v}
        Err(e) => {println!("{e}");return;}
    };
    
    let my_env = MyEnv1::new(v);
    let mut rng = thread_rng();
    let mut renv = REnv::new(my_env);
    let mut preferences = Array1::<f32>::zeros(renv.myenv1.n_actions);
    let mut av_reward : f32 = 0.;
    let mut epsi_av: f32 = 0.05;
    let mut epsi_ch: f32 = 0.0005;
    let starting_it = 100;
    let time = SystemTime::now();
    //dbg!(&my_env.tasks);
    for i in 0..10000{
        let mut state = renv.reset();
        let mut reward = 0.;
        let mut actions_taken = Vec::<usize>::new();
        let mut probas_actions = Vec::<f32>::new();
        while !state.done{
            let n_available_actions = state.available_actions.len();
            let mut probas = Array1::<f32>::zeros(n_available_actions-1);
            for i in 0..n_available_actions-1{
                let action = &state.available_actions[i];
                probas[i] = preferences[action.id];
            }
            let mut action_n = 0;
            if n_available_actions>1{
                probas = probas.mapv_into(|v| v.exp());
                let sum = probas.sum();
                probas = probas.mapv_into(|v| v/sum);
                //dbg!(&probas);
                let dist = WeightedIndex::new(&probas).unwrap();
                action_n = dist.sample(&mut rng);
                actions_taken.push(action_n);
                probas_actions.push(probas[action_n]);
            }
            (state, reward) = renv.step(&state.available_actions[action_n]);
            //dbg!(&state);
        }
        if i> starting_it{
            for id in 0..actions_taken.len(){
                let proba = probas_actions[id];
                preferences[actions_taken[id]]+=(reward-av_reward)*epsi_ch;
            }
        }
        av_reward = av_reward + (reward-av_reward)*epsi_av;

        println!("done, score = {}, time = {}, av_reward = {}", reward, time.elapsed().expect("erreur 1").as_millis(), av_reward);
    }
}

#[derive(Debug)]
struct State{
    timestep : usize,
    done_tasks : Array1<u8>,
    masked_tasks : Array1<u8>,
    available_actions : Vec<Action>,
    done : bool,
}

#[derive(Debug, Clone)]
struct Action{
    task: usize,
    machine: usize,
    operator: usize,
    advance_time : bool,
    id: usize,
}


#[derive(Debug)]
struct REnv{
    myenv1: MyEnv1,
    n_tasks : usize,
    timestep : usize,
}

impl REnv{
    fn new(mut myenv : MyEnv1) -> Self{
        let n_tasks = myenv.tasks.len();
        myenv.reset();
        REnv { myenv1 : myenv,  n_tasks, timestep : 0}
    }

    fn get_state(&mut self) -> State{
        let actions = self.myenv1.get_actions();
        let mut masked_tasks = Array1::<u8>::zeros(self.n_tasks);
        for action in &actions{
            if action.advance_time {
                continue;
            }
            masked_tasks[action.task-1] = 1;
        };
        let mut state = Array1::<u8>::zeros(self.n_tasks);
        for task in &self.myenv1.tasks{
            if task.done{
                state[task.id-1]=1;
            }
        }
        State{ timestep : self.timestep, done_tasks: state, masked_tasks, done: false, available_actions: actions }
    }

    fn reset(&mut self) -> State {
        self.myenv1.reset();
        self.timestep = 0;
        self.get_state()
    }

    fn step(&mut self, action : &Action) -> (State, f32){
        self.timestep+=1;
        let done = self.myenv1.start_task(&action);
        let mut state = self.get_state();
        state.done = done;
        if !done {
            (state, 0.)
        }
        else{
            (state, -(self.myenv1.get_score() as f32))
        }
    }
}

#[derive(Debug)]
struct MyEnv1{
    time: usize,
    operators: Vec<Operator1>,
    machines: Vec<Machine1>,
    jobs: Vec<Job1>, 
    tasks: Vec<Task1>, 
    n_actions : usize,
    parameters : Parameters,
}

impl MyEnv1{
    fn new(myenv : MyEnv) -> Self{
        let mut jobs = Vec::<Job1>::new();
        let mut operators= Vec::<Operator1>::new();
        let mut machines= Vec::<Machine1>::new();
        let mut tasks= Vec::<Task1>::new();
        for machineid in 0..myenv.parameters.size.nb_machines{
            machines.push(Machine1{id: machineid+1,operator:0, used: false, task : 0});

        }
        for operatorid in 0..myenv.parameters.size.nb_operators{
            operators.push(Operator1{id:operatorid+1, task: 0, done:true});

        }

        let mut id = 0;
        for task in myenv.tasks {
            let mut actions = Vec::<Action>::new();
            for machine_task in task.machines{
                for operator in machine_task.operators{
                    actions.push(Action{task : task.task, machine : machine_task.machine, operator, advance_time: false, id});
                    id+=1;
                }
            }
            tasks.push(Task1{id : task.task, machine : 0,job : 0, operator: 0, starting_date : 0, done_date: 0, done: false, options:actions, completion_time : task.processing_time, started : false});
        }
        let n_actions = id;

        for job in myenv.jobs{
            for task in &job.sequence{
                assert_eq!(tasks[task-1].job, 0);
                tasks[task-1].job=job.job;
            }
            jobs.push(Job1{id : job.job, sequence:job.sequence.clone(),  release_date: job.release_date,done_date:0, done: false, weight: job.weight, due_date: job.due_date});
        };

        MyEnv1{time: 0, operators,machines,tasks,jobs, parameters: myenv.parameters, n_actions}
    }

    fn reset(&mut self){
        for task in &mut self.tasks {
            task.starting_date = 0;
            task.done_date = 0;
            task.done = false;
            task.started = false;
            task.machine = 0;
            task.operator = 0;
        }

        for job in &mut self.jobs{
            job.done = false;
            job.done_date = 0;
        }

        for operator in &mut self.operators{
            operator.done = true;
            operator.task = 0;
        }

        for machine in &mut self.machines{
            machine.operator = 0;
            machine.task = 0;
            machine.used = false;
        }
        self.time = 0;
    }

    fn get_actions(&self) -> Vec<Action>{
        let mut actions = Vec::<Action>::with_capacity(self.parameters.size.nb_tasks);
        for job in &self.jobs{
            if job.release_date>self.time{
                continue;
            }
            for taskid in &job.sequence{
                let task = &self.tasks[taskid-1];
                if task.done{
                    continue;
                }
                if task.started{
                    break;
                }
                for action in &task.options{
                    if self.machines[action.machine-1].used{
                        continue;
                    }
                    if self.operators[action.operator-1].done{
                        actions.push(action.clone());
                    }
                }
                break;
            }
        }
        actions.push(Action{task: 0,  machine: 0, operator : 0, advance_time: true, id: 0});
        actions
    }

    fn start_task(&mut self, task: &Action) -> bool{
        if task.advance_time {
            self.advance_time()
        }else{
            self.operators[task.operator-1].done = false;
            self.operators[task.operator-1].task = task.task;
            self.machines[task.machine-1].used=true;
            self.machines[task.machine-1].task = task.task;
            self.machines[task.machine-1].operator= task.operator;
            self.tasks[task.task-1].done_date = self.tasks[task.task-1].completion_time+self.time;
            self.tasks[task.task-1].started= true;
            self.tasks[task.task-1].machine= task.machine;
            self.tasks[task.task-1].operator= task.operator;
            self.tasks[task.task-1].starting_date= self.time;
            false
        }
    }

    fn advance_time(&mut self) -> bool{
        self.time+=1;
        let mut done = true;
        for task in &mut self.tasks{
            if task.done_date == self.time{
                task.done=true;
                let job = &mut self.jobs[task.job-1];
                if *job.sequence.last().unwrap()==task.id{
                    job.done = true;
                    job.done_date = self.time;
                }
                self.machines[task.machine-1].used = false;
                self.operators[task.operator-1].done= true;

            }
            if !task.done{
                done=false}
        }
        done
    }

    fn get_score(&self) -> usize{
        let mut score: usize = 0;
        for job in &self.jobs{
            let u: usize = if job.done_date>job.due_date {1} else{0};
            let tardiness = if job.done_date>job.due_date{job.done_date-job.due_date} else { 0 };
            //dbg!(&job);
            score += job.weight*(cmp::min(job.done_date,self.time)+self.parameters.costs.unit_penalty*u+ self.parameters.costs.tardiness*tardiness);
        }
        score
    }
}

#[derive(Debug)]
struct Machine1{
    id:usize,
    operator:usize,
    used:bool,
    task:usize,
}

#[derive(Debug)]
struct Job1{
    id:usize,
    sequence: Vec<usize>,
    release_date: usize,
    done_date: usize,
    done: bool,
    weight : usize,
    due_date: usize,

}

#[derive(Debug)]
struct Operator1{
    id: usize,
    task: usize,
    done: bool,
}

#[derive(Debug, Clone)]
struct Task1{
    id: usize,
    machine: usize,
    job: usize,
    operator: usize,
    starting_date: usize,
    started : bool,
    completion_time: usize,
    done_date: usize,
    done: bool,
    options: Vec<Action>,
}

#[derive(Serialize, Deserialize, Debug)]
struct Size{ nb_jobs: usize, nb_tasks: usize, nb_machines: usize, nb_operators: usize}

#[derive(Serialize, Deserialize, Debug)]
struct Costs{unit_penalty: usize, tardiness: usize, interim: usize}
#[derive(Serialize, Deserialize, Debug)]
struct Parameters{
    size: Size,
    costs: Costs,
}
#[derive(Serialize, Deserialize, Debug)]
struct MyEnv{
    parameters: Parameters,
    jobs : Vec<Job>,
    tasks : Vec<Task>,
}


#[derive(Serialize, Deserialize, Debug)]
struct Job{
    job:usize,
    sequence: Vec<usize>,
    release_date: usize,
    due_date: usize,
    weight: usize,
}


#[derive(Serialize, Deserialize, Debug)]
struct Task{
    task:usize,
    processing_time:usize,
    machines:Vec<MachineTask>

}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct MachineTask
{
    machine:usize,
    operators:Vec<usize>,
}
