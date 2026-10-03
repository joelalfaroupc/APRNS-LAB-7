import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from ActorCriticNetworks import ActorNetwork, CriticNetwork, copy_target, soft_update
from ReplayBuffer import ReplayBuffer
from helper import episode_reward_plot, video_agent
import numpy as np
from Noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import gymnasium as gym
from gymnasium.wrappers import RecordVideo


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class DDPG:
    """The DDPG Agent."""

    def __init__(self, env, replay_size=1000000, batch_size=32, gamma=0.99, eval_env=None):
        """ Initializes the DDPG method.
        
        Parameters
        ----------
        env: gym.Environment
            The gym environment the agent should learn in.
        replay_size: int
            The size of the replay buffer.
        batch_size: int
            The number of replay buffer entries an optimization step should be performed on.
        gamma: float
            The discount factor.      
        """

        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.shape[0]
        self.env = env
        # The runner supplies a separate environment so evaluation cannot alter training.
        self.eval_env = eval_env if eval_env is not None else env
        self.replay_buffer = ReplayBuffer(replay_size)
        self.batch_size = batch_size
        self.gamma = gamma

        # Initialize Critic network and target network. Should be named self.Critic 
        self.Critic = CriticNetwork(self.obs_dim, self.act_dim).to(device)
        self.Critic_target = CriticNetwork(self.obs_dim, self.act_dim).to(device)
        copy_target(self.Critic_target, self.Critic)

        # Initialize Actor network and its target network. Should be named self.Actor
        self.Actor = ActorNetwork(self.obs_dim, self.act_dim).to(device)
        self.Actor_target = ActorNetwork(self.obs_dim, self.act_dim).to(device)
        copy_target(self.Actor_target, self.Actor)

        # Define the optimizers for the actor and critic networks as proposed in the paper
        self.optim_dqn = optim.Adam(self.Critic.parameters(), lr=0.001, weight_decay=0.01) 
        self.optim_actor = optim.Adam(self.Actor.parameters(), lr=0.0001) 


    def learn(self, timesteps, seed=None, plot=True):
        """Train the agent for timesteps steps inside self.env.
        After every step taken inside the environment observations, rewards, etc. have to be saved inside the replay buffer.
        If there are enough elements already inside the replay buffer (>batch_size), compute MSBE loss and optimize the critic network.

        Parameters
        ----------
        timesteps: int
            Number of timesteps to optimize the critic network.
        """
        if timesteps < 1:
            raise ValueError("timesteps must be positive")
        self.steps_completed = 0
        self.evaluation_steps = []
        all_rewards = []
        episode_rewards = []
        all_rewards_eval = []
        timeexit = timesteps

        # We use here OUNoise instead of Gaussian to add some exploration to the agent. OU noise is a stochastic process
        # that generates a random sample from a Gaussian distribution whose value at time t depends on the previous value
        # x(t) and the time elapsed since the previous value y(t). It helps to explore the environment better than Gaussian noise.
        # This line initializes the noise with mean 0 and sigma 0.15 (see Noise.py file)
        OUNoise =  OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.act_dim))

        obs, _ = self.env.reset(seed=seed)
        if seed is not None and self.eval_env is not self.env:
            self.eval_env.reset(seed=seed + 10000)
        for timestep in range(1, timesteps + 1):

            action = self.choose_action(obs)

            # Here we sample and add the noise to the action to explore the environment. Notice we clip the action
            # between -1 and 1 because the action space is continuous and bounded between -1 and 1.
            epsilon= OUNoise.sample()
            action = np.clip(action + epsilon, -1, 1)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.replay_buffer.put(obs, action, reward, next_obs, terminated, truncated)
            self.steps_completed = timestep
            
            obs = next_obs
            episode_rewards.append(reward)
            
            if terminated or truncated:
                self.evaluation_steps.append(timestep)
                all_rewards_eval.append(self.eval_episodes())
                print('\rTimestep: ', timestep, '/' ,timesteps,' Episode reward: ',np.round(all_rewards_eval[-1]), 'Episode: ', len(all_rewards), 'Mean R', np.mean(all_rewards_eval[-100:]))
                obs, _ = self.env.reset()
                all_rewards.append(sum(episode_rewards))
                episode_rewards = []
                    
            if len(self.replay_buffer) > self.batch_size:
                # Batch is sampled from the replay buffer and containes a list of tuples (s, a, r, s', term, trunc)
                
                # NOTE: ReplayBuffer.get returns separated numpy arrays, not tuples
                obs_batch, action_batch, reward_batch, next_obs_batch, terminated_batch, truncated_batch = self.replay_buffer.get(self.batch_size)
                
                obs_batch = torch.tensor(obs_batch, dtype=torch.float32).to(device)
                action_batch = torch.tensor(action_batch, dtype=torch.float32).to(device)
                reward_batch = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1).to(device)
                next_obs_batch = torch.tensor(next_obs_batch, dtype=torch.float32).to(device)
                terminated_batch = torch.tensor(terminated_batch, dtype=torch.float32).unsqueeze(1).to(device)
                truncated_batch = torch.tensor(truncated_batch, dtype=torch.float32).unsqueeze(1).to(device)
                
                batch = (obs_batch, action_batch, reward_batch, next_obs_batch, terminated_batch, truncated_batch)

                critic_loss = self.compute_critic_loss(batch)
                self.optim_dqn.zero_grad()
                critic_loss.backward()
                self.optim_dqn.step()

                actor_loss = self.compute_actor_loss(batch)
                self.optim_actor.zero_grad()
                actor_loss.backward()
                self.optim_actor.step()
            soft_update(self.Critic_target, self.Critic, tau=0.001)
            soft_update(self.Actor_target, self.Actor, tau=0.001)
            if len(all_rewards_eval) > 10 and np.mean(all_rewards_eval[-5:]) > 220:
                break
        if plot:
            episode_reward_plot(all_rewards, self.steps_completed, window_size=7, step_size=1)
        return all_rewards, all_rewards_eval
    

    def choose_action(self, s):
        # by the OrnsteinUhlenbeckActionNoise in the main loop.
        with torch.no_grad():
            state = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
            action = self.Actor(state)
        return action.cpu().numpy().flatten()


    def compute_critic_loss(self, batch):
        """
        The function computes the critic loss using the Mean Squared Bellman Error (MSBE) calculation.
        
        :param batch: The `batch` parameter is a tuple containing the data for computing the loss.
        :return: the critic loss, which is calculated using the mean squared error (MSE) loss between
        the expected Q-values (q_expected) and the target Q-values (target).
        """
        
        # similar to the DQN loss.
        obs, action, reward, next_obs, terminated, truncated = batch

        with torch.no_grad():
            next_action = self.Actor_target(next_obs)
            target_q = self.Critic_target(next_obs, next_action)
            target_q = reward + (1 - terminated) * self.gamma * target_q

        current_q = self.Critic(obs, action)
        
        return nn.MSELoss()(current_q, target_q)
    

    def compute_actor_loss(self,batch):
        """
        The function `compute_actor_loss` calculates the loss for the actor network 
        
        :param batch: The batch parameter is a tuple containing the data for computing the loss.
        :return: the loss, which is the negative mean of the expected Q-values.
        """
        obs, _, _, _, _, _ = batch
        
        predicted_action = self.Actor(obs)
        actor_loss = -self.Critic(obs, predicted_action).mean()
        
        return actor_loss

  



    def eval_episodes(self,n=3):
        """ Evaluate an agent performing inside a Gym environment. """
        lr=[]
        for episode in range(n):
            tr = 0.0
            obs, _ = self.eval_env.reset()
            while True:
                action = self.choose_action(obs)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                tr += reward
                if terminated or truncated:
                    break
            lr.append(tr)
        return np.mean(lr)




if __name__ == '__main__':
    # Create gym environment
    env = gym.make("LunarLander-v3",continuous=True, render_mode='rgb_array')

    ddpg = DDPG(env,replay_size=1000000, batch_size=64, gamma=0.99)

    ddpg.learn(500000)
    env = RecordVideo(gym.make("LunarLander-v3",continuous=True, render_mode='rgb_array'),'video')    
    video_agent(env, ddpg,n_episodes=5)  
    pass
