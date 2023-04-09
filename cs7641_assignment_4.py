#!/usr/bin/env python
# coding: utf-8

# In[230]:


get_ipython().system('pip install gym')
get_ipython().system('pip install hiive.mdptoolbox')
get_ipython().system('pip install tqdm')

import itertools
from tqdm import tqdm
import gym
import hiive.mdptoolbox as mdptoolbox
from hiive.mdptoolbox.mdp import ValueIteration, PolicyIteration, QLearning
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output, display

# suppress pandas warning
pd.options.mode.chained_assignment = None

# set seed
np.random.seed(0)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[231]:


colors = {
    b'S': 'b',
    b'F': 'w',
    b'H': 'k',
    b'G': 'g'
}

directions = {
            0: '←',
            1: '↓',
            2: '→',
            3: '↑'
}

def plot_lake(env, policy=None, title='Frozen Lake'):
    squares = env.nrow
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, xlim=(-.01, squares+0.01), ylim=(-.01, squares+0.01))
    plt.title(title, fontsize=16, weight='bold', y=1.01)
    for i in range(squares):
        for j in range(squares):
            y = squares - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1, linewidth=1, edgecolor='k')
            p.set_facecolor(colors[env.desc[i,j]])
            ax.add_patch(p)
            
            if policy is not None:
                text = ax.text(x+0.5, y+0.5, directions[policy[i, j]],
                               horizontalalignment='center', size=25, verticalalignment='center',
                               color='k')
            
#     plt.axis('off')
#     plt.savefig('./frozen/' + title + '.png', dpi=400)


# In[232]:


def get_score(env, policy, printInfo=False, episodes=1000):
    misses = 0
    steps_list = []
    for episode in range(episodes):
        observation = env.reset()
        steps=0
        while True:
            action = policy[observation]
            observation, reward, done, _ = env.step(action)
            steps+=1
            if done and reward == 1:
                # print('You have got the Frisbee after {} steps'.format(steps))
                steps_list.append(steps)
                break
            elif done and reward == 0:
                # print("You fell in a hole!")
                misses += 1
                break
    ave_steps = np.mean(steps_list)
    std_steps = np.std(steps_list)
    pct_fail  = (misses/episodes)* 100
    
    if (printInfo):
        print('----------------------------------------------')
        print('You took an average of {:.0f} steps to get the frisbee'.format(ave_steps))
        print('And you fell in the hole {:.2f} % of the times'.format(pct_fail))
        print('----------------------------------------------')
  
    return ave_steps, std_steps, pct_fail


# In[233]:


# def get_policy(env,stateValue, lmbda=0.9):
#     policy = [0 for i in range(env.nS)]
#     for state in range(env.nS):
#         action_values = []
#         for action in range(env.nA):
#             action_value = 0
#             for i in range(len(env.P[state][action])):
#                 prob, next_state, r, _ = env.P[state][action][i]
#                 action_value += prob * (r + lmbda * stateValue[next_state])
#             action_values.append(action_value)
#         best_action = np.argmax(np.asarray(action_values))
#         policy[state] = best_action
#     return policy 


# In[ ]:


size = 4 # or 8 but may take very long for VI to converge


# In[521]:


size = 4 # or 8
if size == 4:
    frozen_map_size = "4x4"
elif size == 8:
    frozen_map_size = "8x8"
else:
    raise Exception('map size needs to be either 4 or 8')

env = gym.make('FrozenLake-v1', map_name= frozen_map_size).unwrapped

env.max_episode_steps=250

# Create transition and reward matrices from OpenAI P matrix
rows = env.nrow
cols = env.ncol
T = np.zeros((4, rows*cols, rows*cols))
R = np.zeros((4, rows*cols, rows*cols))

old_state = np.inf

for square in env.P:
    for action in env.P[square]:
        for i in range(len(env.P[square][action])):
            new_state = env.P[square][action][i][1]
            if new_state == old_state:
                T[action][square][env.P[square][action][i][1]] = T[action][square][old_state] + env.P[square][action][i][0]
                R[action][square][env.P[square][action][i][1]] = R[action][square][old_state] + env.P[square][action][i][2]
            else:
                T[action][square][env.P[square][action][i][1]] = env.P[square][action][i][0]
                R[action][square][env.P[square][action][i][1]] = env.P[square][action][i][2]
            old_state = env.P[square][action][i][1]
            
#print(T)
#print(R)
plot_lake(env)


# In[489]:


def run_value_iteration(t, r, gamma, epsilon, max_iterations):
    test = ValueIteration(t, r, gamma=gamma, epsilon=epsilon, max_iter=max_iterations)
    runs = test.run()
    policy = test.policy
    return runs, policy

def get_policy_stats(data, policy, env, showResults=False):
    policies = data['policy']
    #print(f'policies:{policies}' )
    data['average_steps'], data['steps_stddev'], data['success_pct'] = 0, 0, 0
    
    for i,p in enumerate(policies):
        try:
            pol = list(p)[0]
            steps, steps_stddev, failures = get_score(env, pol, showResults)
        except:
            pol = None
            steps, steps_stddev, failures = 250, 0, 100
        data['average_steps'][i] = steps
        data['steps_stddev'][i]  = steps_stddev
        data['success_pct'][i]   = 100-failures 
    return data

def valueIterationz(t, r, gammas, epsilons, showResults=False, max_iterations=100000):
    t0 = time.time()
    
    # create data structure to save off
    columns = ['gamma', 'epsilon', 'time', 'iterations', 'reward', 'policy', 'mean_rewards', 'max_rewards', 'error']
    data = pd.DataFrame(0.0, index=np.arange(len(gammas)*len(epsilons)), columns=columns)
    
#     print('Gamma,\tEps,\tTime,\tIter,\tReward')
#     print(100*'+')
    
    testNum = 0
    for g in gammas:
        for e in epsilons:
            runs, test_policy = run_value_iteration(t, r, g, e, max_iterations)
            
            Time = runs[-1]['Time']
            iters = runs[-1]['Iteration']
            maxR = runs[-1]['Max V']
            max_rewards, mean_rewards, errors = [], [], []
            max_rewards = [run['Max V'] for run in runs]
            mean_rewards = [run['Mean V'] for run in runs]
            errors = [run['Error'] for run in runs]
                
            policy = np.array(test_policy)
            policy = policy.reshape(size,size)
            
            data.loc[testNum, ['gamma', 'epsilon', 'time', 'iterations', 'reward', 'mean_rewards', 'max_rewards', 'error', 'policy']] = g, e, Time, iters, maxR, {tuple(mean_rewards)}, {tuple(max_rewards)}, {tuple(errors)}, {test_policy}         
            print('%.2f,\t%.0E,\t%.2f,\t%d,\t%f' % (g, e, Time, iters, maxR))
            
            testNum += 1
                
    endTime = time.time() - t0
    print("Time taken: %.2f" %endTime)
    
    data = get_policy_stats(data, policy, env, showResults)
    
    # replace all NaN's
    data.fillna(0, inplace=True)
    
    return data


# In[467]:


gammas   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
epsilons = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1] #[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
vi_data  = valueIterationz(T, R, gammas, epsilons, showResults=False)

interest = ['gamma', 'epsilon', 'time', 'iterations', 'reward']
df = vi_data[interest]


# In[468]:


def plot_performance(results_df, values, performance_col, x_axis_name, title):
    x = values
    y = []
    for g in values:
        y.append(results_df.loc[values == g][performance_col].mean())
    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(6,4))
    ax = sns.lineplot(x=x, y=y)
    ax.set_title(title)
    ax.set_xlabel(x_axis_name)
    ax.set_ylabel(performance_col)
    plt.show()


# In[469]:


plot_performance(vi_data,vi_data['gamma'], 'average_steps', 'gamma', 'Average Steps vs Gamma')


# In[470]:


plot_performance(vi_data,vi_data['gamma'], 'success_pct', 'gamma', 'Success % vs Gamma')


# In[471]:


plot_performance(vi_data,vi_data['gamma'], 'reward', 'gamma', 'Reward vs Gamma')


# In[472]:


plot_performance(vi_data,vi_data['gamma'], 'time', 'gamma', 'Time to Converge vs Gamma')


# In[473]:


plot_performance(vi_data,vi_data['epsilon'], 'average_steps', 'epsilon', 'Average Steps vs Epsilon')


# In[474]:


plot_performance(vi_data,vi_data['epsilon'], 'success_pct', 'epsilon', 'Success % vs epsilon')


# In[475]:


plot_performance(vi_data,vi_data['epsilon'], 'reward', 'epsilon', 'Reward vs epsilon')


# In[476]:


plot_performance(vi_data,vi_data['epsilon'], 'time', 'epsilon', 'Convergence time vs epsilon')


# In[477]:


plot_performance(vi_data, vi_data['iterations'], 'success_pct', 'Iterations', 'Success % vs Iterations')


# In[478]:


def plot_best_policy(results_df, env, rows, cols, env_name = 'Frozen Lake: ' + frozen_map_size, mdp_algo = 'VI'):
    best_run = results_df['success_pct'].idxmax()
    best_policy = np.array(list(results_df['policy'][best_run]))[0].reshape(rows, cols)
    if mdp_algo == 'VI':
        title = env_name + ' ' + mdp_algo  + ' Optimal Policy (Success = {:.2f}, Gamma = {:.2f}, Epsilon = {})'.format(results_df['success_pct'][best_run], results_df['gamma'][best_run], results_df['epsilon'][best_run])
    elif mdp_algo == 'PI':
        title = env_name + ' ' + mdp_algo + ' Optimal Policy (Success = {:.2f}, Gamma = {:.2f})'.format(results_df['success_pct'][best_run], results_df['gamma'][best_run])
    elif mdp_algo == 'QL':
        title = env_name + ' ' + mdp_algo  + ' Best Result:\n\tSuccess = %.2f\n\tGamma = %.2f,\n\tAlpha = %.2f,\n\tAlpha Decay: %.3f,\n\tEpsilon Decay: %.3f,\n\tIterations: %.1E' % (results_df['success_pct'].max(), results_df['gamma'][best_run], results_df['alpha'][best_run], results_df['alpha_decay'][best_run], results_df['epsilon_decay'][best_run], results_df['iterations'][best_run])
    
    plot_lake(env, best_policy, title)


# In[479]:


plot_best_policy(vi_data, env, size, size)


# In[558]:


def get_policy_stats(data, policy, env, showResults=False):
    policies = data['policy']
    print(f'policies:{policies}')
    #print(f'policies:{policies}' )
    data['average_steps'], data['steps_stddev'], data['success_pct'] = 0, 0, 0
    
    for i,p in enumerate(policies):
        print(p)
        pol = list(p)[0]
        steps, steps_stddev, failures = get_score(env, pol, showResults)
        data['average_steps'][i] = steps
        data['steps_stddev'][i]  = steps_stddev
        data['success_pct'][i]   = 100-failures 
    return data



def run_policy_iteration(t, r, gamma, max_iterations = 100000, eval_type="matrix"):
    test =  PolicyIteration(t, r, gamma=gamma, max_iter=max_iterations, eval_type=eval_type)
    runs = test.run()
 #   policy = test.policy
    max_value  = runs[-1]['Max V']
#    print(r)#     print(test.iter)
#     print(runs[-1]['Iteration'])
    return runs, test.policy, test.time, test.iter, max_value



def policyIterationz(t, r, gammas, showResults=False, max_iterations=100000, eval_type="matrix", is_grid = True, tsize = 4):
    t0 = time.time()
    
    columns = ['gamma', 'time', 'iterations', 'reward', 'mean_rewards', 'max_rewards', 'error', 'policy']
    data = pd.DataFrame(0.0, index=np.arange(len(gammas)*len(epsilons)), columns=columns)
    
    testNum = 0
    for g in gammas:
        print(f'gamma:{g}')
        try:
            runs, test_policy, test_time, test_iterations, max_reward = run_policy_iteration(t, r, g, max_iterations, eval_type)
            print(f'policy:{test_policy}')
            max_rewards, mean_rewards, errors = [], [], []
            max_rewards = [run['Max V'] for run in runs]
            mean_rewards = [run['Mean V'] for run in runs]
            errors = [run['Error'] for run in runs]

            policy = np.array(test_policy)
            policy = policy.reshape(tsize,tsize)

            data.loc[testNum, columns] = g, test_time, test_iterations, max_reward, {tuple(mean_rewards)}, {tuple(max_rewards)}, {tuple(errors)}, {test_policy}         
            print('%.2f,\t%.2f,\t%d,\t%f' % (g, test_time, test_iterations, max_reward))
        except:
            print("error occured")
            pass
        testNum += 1
    #print(data)

    endTime = time.time() - t0
    print("Time taken: %.2f" %endTime)
    
    data = data[data['policy'] != 0]
    
    data = get_policy_stats(data, policy, env, showResults)
    
    # replace all NaN's
    data.fillna(0, inplace=True)
    
    return data


# In[548]:


def policyIteration(t, r, gammas, showResults=False, max_iterations=100000):
    t0 = time.time()
    
    # create data structure to save off
    columns = ['gamma', 'epsilon', 'time', 'iterations', 'reward', 'average_steps', 'steps_stddev', 'success_pct', 'policy', 'mean_rewards', 'max_rewards', 'error']
    data = pd.DataFrame(0.0, index=np.arange(len(gammas)), columns=columns)
    
    print('gamma,\ttime,\titer,\treward')
    print(80*'_')
    
    testnum = 0
    for g in gammas:
        test = PolicyIteration(t, r, gamma=g, max_iter=max_iterations, eval_type="matrix") # eval_type="iterative"
        
        runs  = test.run()
        Time  = test.time
        iters = test.iter
        maxr  = runs[-1]['Max V']
                
        max_rewards, mean_rewards, errors = [], [], []
        for run in runs:
            max_rewards.append(run['Max V'])
            mean_rewards.append(run['Mean V'])
            errors.append(run['Error'])
            
        
        print(f'test policy:{test.policy}')
        policy = np.array(test.policy)
        policy = policy.reshape(4,4)
        
        data['gamma'][testnum]        = g
        data['time'][testnum]         = Time
        data['iterations'][testnum]   = iters
        data['reward'][testnum]       = maxr
        data['mean_rewards'][testnum] = {tuple(mean_rewards)}
        data['max_rewards'][testnum]  = {tuple(max_rewards)}
        data['error'][testnum]        = {tuple(errors)}
        data['policy'][testnum]       = {test.policy}
        
        print('%.2f,\t%.2f,\t%d,\t%f' % (g, Time, iters, maxr))
        
        if showResults:
            title = 'frozenlake_pi_' + str(rows) + 'x' + str(cols) + '_g' + str(g)
            plot_lake(env, policy, title)
        
        testnum = testnum + 1
            
    endTime = time.time() - t0
    print('Time taken: %.2f' % endTime)
    
    # see differences in policy
    policies = data['policy']
    print(f'policies:{policies}')
    
    for i,p in enumerate(policies):
        pol = list(p)[0]
        print(pol)
        steps, steps_stddev, failures = get_score(env, pol, showResults)
        data['average_steps'][i] = steps
        data['steps_stddev'][i]  = steps_stddev
        data['success_pct'][i]   = 100-failures      
        
    # replace all nan's
    data.fillna(0, inplace=True)
    data.head()
        
    return data


# In[559]:


gammas   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
pi_data  = policyIterationz(T, R, gammas, showResults=False, tsize = 4)


# In[560]:


plot_performance(pi_data, pi_data['gamma'], 'average_steps', 'gamma', 'Average Steps vs Gamma')


# In[561]:


plot_performance(pi_data, pi_data['gamma'], 'success_pct', 'gamma', 'Success % vs Gamma')


# In[562]:


plot_performance(pi_data, pi_data['gamma'], 'time', 'gamma', 'Time taken to converge vs Gamma')


# In[563]:


plot_performance(pi_data, pi_data['gamma'], 'reward', 'gamma', 'Rewards vs Gamma')


# In[564]:


plot_performance(pi_data, pi_data['gamma'], 'iterations', 'gamma', 'Iterations vs Gamma')


# In[565]:


plot_performance(pi_data, pi_data['success_pct'], 'success_pct', 'iterations', 'Success_pct vs Iterations')


# In[566]:


plot_best_policy(pi_data, env, size, size,  mdp_algo = 'PI')


# In[147]:


get_ipython().system(' pip install tqdm')
from tqdm import tqdm
import itertools


# In[567]:



def run_q_learning(t, r, gamma, alpha, alpha_decay, epsilon_decay, n_iter):
    test =  QLearning(t, r, gamma, alpha, alpha_decay, epsilon_decay, n_iter)
    runs = test.run()
    policy = test.policy
    test_time = runs[-1]['Time']
    test_iter = runs[-1]['Iteration']
    max_value  = runs[-1]['Max V']
    return runs, policy, test_time, test_iter, max_value


def Qlearningz(t, r, gammas, alphas, alpha_decays=[0.99], epsilon_decays=[0.99], n_iterations=[10000000], showResults=False):
    t0 = time.time()
    
    columns = ['gamma', 'alpha', 'alpha_decay', 'epsilon_decay', 'time', 'iterations', 'reward', 'mean_rewards', 'max_rewards', 'error', 'policy']
    total_tests = len(gammas) * len(alphas) * len(alpha_decays) * len(epsilon_decays) * len(n_iterations)
    data = pd.DataFrame(0.0, index=np.arange(total_tests), columns=columns)
    
    testNum = 0
    for i, config in tqdm(enumerate(itertools.product(gammas, alphas, alpha_decays, epsilon_decays, n_iterations)), total=total_tests):
        g, a, a_decay, e_decay, n = config
        runs, test_policy, test_time, test_iterations, max_reward = run_q_learning(t, r, gamma=g, alpha=a, alpha_decay=a_decay, epsilon_decay=e_decay, n_iter=n)
        max_rewards, mean_rewards, errors = [], [], []
        max_rewards = [run['Max V'] for run in runs]
        mean_rewards = [run['Mean V'] for run in runs]
        errors = [run['Error'] for run in runs]

        policy = np.array(test_policy)
        policy = policy.reshape(size,size)

        data.loc[testNum, columns] = g, a, a_decay, e_decay, test_time, test_iterations, max_reward, {tuple(mean_rewards)}, {tuple(max_rewards)}, {tuple(errors)}, {test_policy}         
        print('%.2f,\t%.2f,\t%.2f,\t%d,\t%f' % (g, a, test_time, test_iterations, max_reward))

        testNum += 1

    endTime = time.time() - t0
    print("Time taken: %.2f" %endTime)
    
    data = data[data['policy'] != 0]
    print(data['policy'])
    
    data = get_policy_stats(data, policy, env, showResults)
    
    # replace all NaN's
    data.fillna(0, inplace=True)
    
    return data


# In[354]:


plot_performance(qlearning_data, qlearning_data['gamma'], 'average_steps', 'gamma', 'Average Steps vs Gamma')


# In[355]:


plot_performance(qlearning_data, qlearning_data['gamma'], 'success_pct', 'gamma', 'Mean Success % vs Gamma')


# In[356]:


plot_performance(qlearning_data, qlearning_data['epsilon_decay'], 'average_steps', 'epsilon_decay', 'Average Steps vs epsilon_decays')


# In[357]:


plot_performance(qlearning_data, qlearning_data['epsilon_decay'], 'success_pct', 'epsilon_decay', 'Success % vs epsilon_decays')


# In[ ]:





# In[366]:


plot_best_policy(qlearning_data, env, size, size, 'Frozen Lake 4x4',  mdp_algo = 'QL')


# # Forest

# In[571]:


policy_colors = {
    0: 'g', # stands for Wait
    1: 'k'  # stands for Cut
}

policy_labels = {
    0: 'W', # stands for Wait
    1: 'C', # stands for Cut
}

def plot_forest_management_policy(policy, title='Forest Management Policy'):
    # Define the dimensions of the policy grid
    rows = 25
    cols = 25
    
    # Reshape the policy array to 2D grid
    policy = np.array(list(policy)).reshape(rows, cols)
    
    # Create a figure to plot the policy on
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, xlim=(-.01, cols+0.01), ylim=(-.01, rows+0.01))
    
    # Add a title to the plot
    plt.title(title, fontsize=16, weight='bold', y=1.01)
    
    # Loop through each cell of the policy grid
    for i in range(rows):
        for j in range(cols):
            # Calculate the coordinates of the cell
            y = rows - i - 1
            x = j
            
            # Create a rectangle to represent the cell
            rect = plt.Rectangle([x, y], 1, 1, linewidth=1, edgecolor='k')
            
            # Set the face color of the rectangle to represent the policy
            rect.set_facecolor(policy_colors[policy[i, j]])
            
            # Add the rectangle to the plot
            ax.add_patch(rect)
            
            # Add the policy label to the cell
            label = policy_labels[policy[i, j]]
            text = ax.text(x+0.5, y+0.5, label, horizontalalignment='center', size=10, verticalalignment='center', color='w')


# In[572]:


import hiive.mdptoolbox.example
forest_size = 25
T,R = mdptoolbox.example.forest(S=forest_size ** 2)
size = forest_size


# In[573]:


# gammas   = [0.1, 0.3, 0.6, 0.9, 0.9999999]
# epsilons = [1e-2, 1e-3, 1e-8, 1e-12]
gammas   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
epsilons = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
vi_data  = valueIterationz(T, R, gammas, epsilons, showResults=False)


# In[574]:


plot_performance(vi_data,vi_data['gamma'], 'average_steps', 'gamma', 'Average Steps vs Gamma')


# In[575]:


plot_performance(vi_data,vi_data['gamma'], 'success_pct', 'gamma', 'Success % vs Gamma')


# In[576]:


plot_performance(vi_data,vi_data['epsilon'], 'average_steps', 'epsilon', 'Average Steps vs Epsilon')


# In[577]:


plot_performance(vi_data,vi_data['epsilon'], 'success_pct', 'epsilon', 'Success % vs Epsilon')


# In[578]:


plot_performance(vi_data,vi_data['iterations'], 'success_pct', 'iterations', 'Success % vs iterations')


# In[579]:


def plot_best_policy_forest(data, title):
    best_run_index = data['reward'].idxmax()
    best_policy = data['policy'][best_run_index]
    plot_forest_management_policy(best_policy, title)


# In[594]:


# find the highest score
plot_best_policy_forest(vi_data, 'Forest Management Best Policy')
print('Best Result:\n\tReward = %.2f\n\tGamma = %.7f\n\tEpsilon = %.E' % (vi_data['reward'].max(), vi_data['gamma'][vi_data['reward'].idxmax()], vi_data['epsilon'][vi_data['success_pct'].argmax()]))


# In[581]:


pi_data  = policyIterationz(T, R, gammas, showResults=False, tsize = 25)
pi_data[pi_data['success_pct']>0]


# In[582]:


plot_performance(pi_data,pi_data['gamma'], 'average_steps', 'gamma', 'Average Steps vs Gamma')


# In[583]:


plot_performance(pi_data,pi_data['gamma'], 'success_pct', 'gamma', 'Success % vs Gamma')


# In[584]:


plot_performance(pi_data,pi_data['iterations'], 'success_pct', 'iterations', 'Success % vs iterations')


# In[591]:


plot_best_policy_forest(pi_data, 'Forest Management Best Policy')
print('Best Result:\n\tReward = %.2f\n\tGamma = %.7f\n\t' % (vi_data['reward'].max(), vi_data['gamma'][vi_data['success_pct'].argmax()]))


# In[586]:


gammas   = [0.1, 0.5, 0.9, 0.99]
alphas   = [0.01, 0.1, 0.2]
alpha_decays = [0.9, 0.999]
epsilon_decays = [0.9, 0.999]
iterations = [1e3, 1e4, 1e5]


qlearning_data  = Qlearningz(T, R, gammas, alphas, alpha_decays=alpha_decays, epsilon_decays=epsilon_decays, n_iterations=iterations, showResults=True)


qlearning_data[qlearning_data['success_pct'] > 0]


# In[587]:


plot_performance(qlearning_data,qlearning_data['gamma'], 'average_steps', 'gamma', 'Average Steps vs Gamma')


# In[588]:


plot_performance(qlearning_data,qlearning_data['gamma'], 'success_pct', 'gamma', 'Success % vs Gamma')


# In[589]:


plot_performance(qlearning_data,qlearning_data['epsilon_decay'], 'average_steps', 'epsilon_decay', 'Average Steps vs epsilon_decay')


# In[590]:


# find the highest score
plot_best_policy_forest(qlearning_data, 'Forest Management Best Policy')



best_run_index = qlearning_data['reward'].idxmax()
best_policy = qlearning_data['policy'][best_run_index]
print('Best Result:\n\tReward = %.2f\n\tGamma = %.2f,\n\tAlpha = %.2f,\n\tAlpha Decay: %.3f,\n\tEpsilon Decay: %.3f,\n\tIterations: %.1E'
% (qlearning_data['reward'].max(), qlearning_data['gamma'][best_run_index], qlearning_data['alpha'][best_run_index], qlearning_data['alpha_decay'][best_run_index], qlearning_data['epsilon_decay'][best_run_index], qlearning_data['iterations'][best_run_index]))


# In[ ]:




