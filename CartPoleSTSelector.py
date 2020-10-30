import gym
import numpy as np
import tensorflow as tf
from scipy.stats import norm
from garage import wrap_experiment
import scipy
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.envs import normalize
import sys
from garage.experiment.experiment import ExperimentContext


action_probs_global = []
selected_task = []
testing_inputs_total = []

def UssefulnessMeasure(source_task, EPOCS = 5, NUMBER_EPISODES = 10, test_number= 0):
        

    validationTasks = [] #VALIDATION TASKS

    isUssefulEntropy_Final = False

 
    final_entrop_list = []

    np.set_printoptions(threshold=sys.maxsize)

    @wrap_experiment(log_dir="./LoadDir/CartPole/TaskST"+str(source_task),snapshot_mode="none",snapshot_gap= 0,use_existing_dir=True,name="TaskSTTask"+str(source_task))
    def evaluate_target_tasks(ctxt=None,targetTask=0, STATE_SAMPLEs = 100):
        isUssefulEntropy = False

        entropy_start = 0
        entropy_end = 0
        entropy_list = np.zeros(NUMBER_EPISODES)
        for _ in range(EPOCS):
            action_probs = []
            tf.compat.v1.reset_default_graph()
            with tf.compat.v1.Session() as sess:
                
                with LocalTFRunner(snapshot_config=ctxt,sess=sess) as runner:
                    
                    saved_dir = "./Saved_Models/CartPole/Task"+str(source_task)

                    state_samples = []

                    # trainer = 1
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
                        _,infos = runner._policy.get_action(state)
                        
                        act_prob = infos["prob"]
                        act_prob = np.array(act_prob)
                        act_prob = np.reshape(act_prob,-1)
                        act_prob[act_prob < 0.0001] = 0.0001

                      
                        ent = scipy.stats.entropy(act_prob)

                        action_probs.append(ent)
                                
                    
                    runner.resume(n_epochs=runner._stats.total_epoch+(NUMBER_EPISODES),batch_size=2000)
                  
                        
                    for st in state_samples:
                        _,infos = runner._policy.get_action(st)
                        act_prob = infos["prob"]
                        act_prob = np.array(act_prob)
                        act_prob = np.reshape(act_prob,-1)
                        act_prob[act_prob < 0.0001] = 0.0001
                        ent = scipy.stats.entropy(act_prob)
         
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
def TaskSelection(dif_value, task = 0,number_test_inputs = 100,test_number=0):
    
    global averageError
    isDifferent = True
    isUssefulEntropy = None
    difference_acceptance = dif_value 
    if len(selected_task) != 0:
        for key in selected_task:
            deltakl = 0
            
            for j in range(number_test_inputs):
                try:
                    deltakl += abs(scipy.stats.entropy(action_probs_global[key-starting_task][j],action_probs_global[task-starting_task][j]))
                    
                except:
                    print("Error on j {} key {}".format(j,key))
                    sys.exit()
            deltakl = deltakl/(number_test_inputs)
            
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
 
    testing_inputs = [] #STATE SAMPLES
    
    action_probs = []             
    @wrap_experiment(log_dir="./LoadDir/CartPole/TestST"+str(task),snapshot_mode="none",snapshot_gap= 0,use_existing_dir=True,name="TestSTTask"+str(task))
    def prob_exp(ctxt):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            
            with LocalTFRunner(snapshot_config=ctxt,sess=sess) as runner:
                saved_dir = "./LoadDir/CartPole/Task"+str(task)
                
                runner.restore(from_dir=saved_dir)

                for k in range(len(testing_inputs)):



                    state = testing_inputs[k]
                    _,infos = runner._policy.get_action(state)
             
                    
                    act_prob = infos["prob"]
              
                    act_prob = np.array(act_prob)
                    act_prob = np.reshape(act_prob,-1)
                    act_prob[act_prob < 0.0001] = 0.0001
                    action_probs.append(act_prob)
              
            sess.close()
            
            
    prob_exp()
    action_probs_global.append(action_probs)
    
def main(load_model = False):
    
    test_number = 0

    taskRange = 60
    starting_task = 0
    dif_value = 0.25
    number_test_inputs = 0
   
    acepted_tasks = []
    for t in range(starting_task, taskRange):
                
        extract_action_prob(test_number,t,number_test_inputs=number_test_inputs)
        saved = TaskSelection(dif_value,task=t,test_number=test_number,number_test_inputs=number_test_inputs)
        
        if saved:
            acepted_tasks.append(t)
    print("SELECTED TASKS:")
    print(acepted_tasks)

if __name__ == '__main__':
    main(True)
