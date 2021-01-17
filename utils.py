from unityagents import UnityEnvironment
import numpy as np
from collections import deque
from tqdm.notebook import tqdm
import torch
import matplotlib.pyplot as plt


def plot_scores(scores, scores_means, execution_info):
    
    ## Plot the scores
    fig = plt.figure(figsize=(20,10))
    
    for key in execution_info:
        print(f'{key}: {execution_info[key]}')

    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.plot(np.arange(len(scores_means)), scores_means)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend(('Score', 'Mean'), fontsize='xx-large')

    plt.show()


def train(agent,
          env,
          brain_name,
          n_agents,
          n_episodes=10000,
          max_t=10000,
          print_every=10,
          win_condition=0.5):
    
    """
    
    Continious Control using DDPG.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        print_every (int): how many episodes before printing scores
        n_agents (int): how many arms are in the environment
    
    """    

    scores = []
    scores_mean = []
    scores_window = deque(maxlen=100) # Score last 100 scores
    
    for i_episode in tqdm(range(1, n_episodes+1)):
        
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        states = env_info.vector_observations             # get the current state (for each agent)
        score = np.zeros(n_agents)                     # initialize the score (for each agent)
        
        for t in range(max_t):
            actions = agent.act(states)                   # consult agent for actions
            env_info = env.step(actions)[brain_name]      # take a step in the env
            next_states = env_info.vector_observations    # get next state (for each agent)
            rewards = env_info.rewards                    # get reward (for each agent)
            dones = env_info.local_done                   # see if episode finished
            
            # take a learning step
            agent.step(states, actions, rewards, next_states, dones)  
            
            score += env_info.rewards                    # update the score (for each agent)
            states = next_states                          # roll over states to next time step
            if np.any(dones):                             # exit loop if episode finished
                break

        scores.append(np.mean(score))
        scores_window.append(np.mean(score))
        scores_mean.append(np.mean(scores_window))
        
        # Print on print_every condition
        if i_episode % print_every == 0:
            print('\rEpisode {}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, np.mean(score), np.mean(scores_window)))    
        
        # Winning condition + save model parameters    
        if np.mean(scores_window) >= win_condition:
            print('\nEnvironment solved in {:d} episodes!\t Score: {:.2f} \tAverage Score: {:.2f}'.format(i_episode, np.mean(score), np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break

    execution_info = {'last_score': scores.pop(),
                      'solved_in': i_episode,
                      'last_100_avg': np.mean(scores_window)}
    
    return scores, scores_mean, execution_info 