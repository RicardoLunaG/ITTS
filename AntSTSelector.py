import gym
import numpy as np
import tensorflow as tf
from scipy.stats import norm
from garage import wrap_experiment
import scipy
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.envs import normalize
from garage.experiment.experiment import ExperimentContext
from gym.envs.mujoco.ant_rand_vel import AntVelEnv

A_DIM = 8

action_probs_global = []
selected_task = []
testing_inputs_total = []

def UssefulnessMeasure(source_task, EPOCS = 5, NUMBER_EPISODES = 10, test_number= 0):
    
    a_h = 1
    a_l = -1
    validationTasks = [] #Validation Tasks

    isUssefulEntropy_Final = False
    
    x = np.arange(a_l, a_h, 0.001)
    final_entrop_list = []
  

    @wrap_experiment(log_dir="./LoadDir/Ant/Task"+str(source_task),snapshot_mode="none",snapshot_gap= 0,use_existing_dir=False,name="Task"+str(source_task))
    def evaluate_target_tasks(ctxt=None,targetTask=0,STATE_SAMPLEs = 100):
        isUssefulEntropy = False
        final_average_reward = 0
        entropy_start = 0
        entropy_end = 0
        entropy_list = np.zeros(NUMBER_EPISODES)
        for _ in range(EPOCS):
            action_probs = []
            tf.compat.v1.reset_default_graph()
            with tf.compat.v1.Session() as sess:
                
                with LocalTFRunner(snapshot_config=ctxt,sess=sess) as runner:
                    
                    saved_dir = "./SavedDir/Task"+str(source_task)
       
                    state_samples = []

                    env = validationTasks[targetTask]
                    env = TfEnv(normalize(env))
                    runner.restore(from_dir=saved_dir,env = env)

                    itera = 0
                    
                    
                    itera += 1
                    s = env.reset()
              
                    for _ in range(STATE_SAMPLEs):
                        a = runner.policy.get_action(s)
                        state_samples.append(s)
                        s, _, done, _ = env.step(a)
                        if done:
                            s = env.reset()
                
               
                    for state in state_samples:
                        action,infos = runner._policy.get_action(state)
                      
                        mu = infos["mean"]
                        sigma = infos["log_std"]

                        ent = 0
                        for m_,sig in zip(mu,sigma):
                           
                            q_pdf = norm.pdf(x, m_, np.sqrt(abs(sig)))
                            q_pdf[q_pdf < 0.0001] = 0.0001
                           
                            ent += scipy.stats.entropy(q_pdf)
                        ent = ent/len(mu)
                        action_probs.append(ent)
                                
                    
                    runner.resume(n_epochs=runner._stats.total_epoch+(NUMBER_EPISODES),batch_size=1000)
                    
                        
                    for st in state_samples:
                        action,infos = runner._policy.get_action(st)
                        mu = infos["mean"]
                        sigma = infos["log_std"]
                        ent = 0
                        for m_,sig in zip(mu,sigma):
                            q_pdf = norm.pdf(x, m_, np.sqrt(abs(sig)))
                            q_pdf[q_pdf < 0.0001] = 0.0001
                            ent += scipy.stats.entropy(q_pdf)
                        ent = ent/len(mu)
                        action_probs.append(ent)
                 
                    action_probs = np.array(action_probs)
                    
                    half = len(action_probs)//2
                   
                    action_probs1 = action_probs[:half]
                   
                    action_probs2 = action_probs[half:]
                   
                    
                    entrop1 = np.average(action_probs1)
                    entrop2 = np.average(action_probs2)
                    
                    entropy_start += entrop1
                    entropy_end += entrop2
            sess.close()
            
        entropy_start = entropy_start/EPOCS
        entropy_end = entropy_end/EPOCS
        entropy_list = np.divide(entropy_list,EPOCS)
        final_entrop_list.append(entropy_list)

        if entropy_start >= entropy_end:
            isUssefulEntropy = True
        return isUssefulEntropy
        
    for target in range(len(validationTasks)):
        useful = evaluate_target_tasks(targetTask=target)
        isUssefulEntropy_Final = useful
        if useful == True:
            break

        
    
    return isUssefulEntropy_Final

averageError = 0
def TaskSelection(dif_value, task = 0,number_test_inputs = 100,test_number=0, starting_task = 0):
    
    global averageError
    isDifferent = True
    isUssefulEntropy = None
    difference_acceptance = dif_value 
    if len(selected_task) != 0:
        for key in selected_task:
            deltakl = 0
            
            for j in range(number_test_inputs*A_DIM):
                deltakl += abs(scipy.stats.entropy(action_probs_global[key][j],action_probs_global[task-1][j]))
           
            deltakl = deltakl/(number_test_inputs*A_DIM)
            
            if deltakl < difference_acceptance: 
                
                isDifferent = False

        if isDifferent:
            isUssefulEntropy = UssefulnessMeasure(source_task=task,test_number=test_number)
            
            if isUssefulEntropy:
                selected_task.append(task)
                print("Task {} Acepted".format(task))
    else:
        isUssefulEntropy = UssefulnessMeasure(source_task=task,test_number=test_number)
        
        if isUssefulEntropy:
            selected_task.append(task)
            print("Task {} Acepted".format(task))
    
    return isDifferent and isUssefulEntropy

def extract_action_prob(test_number, task = 0,number_test_inputs=20):

    a_h = 1
    a_l = -1

    testing_inputs = [] #STATE SAMPLES
    
    action_probs = []             
    @wrap_experiment(log_dir="./LoadDir/Ant/Task"+str(task),snapshot_mode="none",snapshot_gap= 0,use_existing_dir=False,name="Task"+str(task))
    def prob_exp(ctxt):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            
            with LocalTFRunner(snapshot_config=ctxt,sess=sess) as runner:
                saved_dir = "./LoadDir/Ant/Task"+str(task)
                
                runner.restore(from_dir=saved_dir)
                
                x = np.arange(a_l, a_h, 0.001)
                for k in range(len(testing_inputs)):
                    state = testing_inputs[k]
                    action,infos = runner._policy.get_action(state)
                    mu = infos["mean"]
                    sigma = infos["log_std"]
                    for m_,sig in zip(mu,sigma):
                        q_pdf = norm.pdf(x, m_, np.sqrt(abs(sig)))
                        q_pdf[q_pdf < 0.0001] = 0.0001
                        action_probs.append(q_pdf)
            sess.close()
            
    prob_exp()
    action_probs_global.append(action_probs)
    
def main(load_model = False):
    
    test_number = 0

    taskRange = 40
    starting_task = 1
    dif_value = 0.15
    number_test_inputs = 100
    acepted_tasks = []
    for t in range(starting_task, taskRange):
        
        
        extract_action_prob(test_number,t,number_test_inputs=number_test_inputs)
        saved = TaskSelection(dif_value,task=t,test_number=test_number,number_test_inputs=number_test_inputs,starting_task=starting_task)
        
        if saved:
            acepted_tasks.append(t)
    print("SELECTED TASKS:")
    print(acepted_tasks)

if __name__ == '__main__':
    main(True)
