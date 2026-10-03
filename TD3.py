import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from ActorCriticNetworks import ActorNetwork, CriticNetwork, copy_target, soft_update
from ReplayBuffer import ReplayBuffer
from helper import episode_reward_plot, video_agent
import numpy as np
from Noise import NormalActionNoise
from gymnasium.wrappers import RecordVideo

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TD3:
    """The TD3 Agent."""

    def __init__(self, env, replay_size=1000000, batch_size=100, gamma=0.99, 
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2, eval_env=None):
        """ Initializes the TD3 method. """

        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.shape[0]
        self.env = env
        # The runner supplies a separate environment so evaluation cannot alter training.
        self.eval_env = eval_env if eval_env is not None else env
        self.replay_buffer = ReplayBuffer(replay_size)
        self.batch_size = batch_size
        self.gamma = gamma
        
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.Critic1 = CriticNetwork(self.obs_dim, self.act_dim).to(device)
        self.Critic1_target = CriticNetwork(self.obs_dim, self.act_dim).to(device)
        copy_target(self.Critic1_target, self.Critic1)

        self.Critic2 = CriticNetwork(self.obs_dim, self.act_dim).to(device)
        self.Critic2_target = CriticNetwork(self.obs_dim, self.act_dim).to(device)
        copy_target(self.Critic2_target, self.Critic2)

        self.Actor = ActorNetwork(self.obs_dim, self.act_dim).to(device)
        self.Actor_target = ActorNetwork(self.obs_dim, self.act_dim).to(device)
        copy_target(self.Actor_target, self.Actor)

        self.optim_critic = optim.Adam(list(self.Critic1.parameters()) + list(self.Critic2.parameters()), lr=0.001)
        self.optim_actor = optim.Adam(self.Actor.parameters(), lr=0.001) 


    def learn(self, timesteps, seed=None, plot=True):
        """Train the agent for timesteps steps inside self.env."""
        if timesteps < 1:
            raise ValueError("timesteps must be positive")
        self.steps_completed = 0
        self.evaluation_steps = []
        all_rewards = []
        episode_rewards = []
        all_rewards_eval = []

        ExplorationNoise = NormalActionNoise(mean=0.0, sigma=0.1)

        obs, _ = self.env.reset(seed=seed)
        if seed is not None and self.eval_env is not self.env:
            self.eval_env.reset(seed=seed + 10000)
        total_it = 0

        for timestep in range(1, timesteps + 1):
            total_it += 1
            
            action = self.choose_action(obs)
            
            noise = ExplorationNoise.sample()
            action = np.clip(action + noise, -1, 1)

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
                obs_batch, action_batch, reward_batch, next_obs_batch, terminated_batch, truncated_batch = self.replay_buffer.get(self.batch_size)
                
                obs_batch = torch.tensor(obs_batch, dtype=torch.float32).to(device)
                action_batch = torch.tensor(action_batch, dtype=torch.float32).to(device)
                reward_batch = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1).to(device)
                next_obs_batch = torch.tensor(next_obs_batch, dtype=torch.float32).to(device)
                terminated_batch = torch.tensor(terminated_batch, dtype=torch.float32).unsqueeze(1).to(device)
                
                batch = (obs_batch, action_batch, reward_batch, next_obs_batch, terminated_batch)

                critic_loss = self.compute_critic_loss(batch)
                self.optim_critic.zero_grad()
                critic_loss.backward()
                self.optim_critic.step()

                if total_it % self.policy_freq == 0:
                    actor_loss = self.compute_actor_loss(batch)
                    self.optim_actor.zero_grad()
                    actor_loss.backward()
                    self.optim_actor.step()

                    soft_update(self.Critic1_target, self.Critic1, tau=0.005)
                    soft_update(self.Critic2_target, self.Critic2, tau=0.005)
                    soft_update(self.Actor_target, self.Actor, tau=0.005)

            if len(all_rewards_eval) > 10 and np.mean(all_rewards_eval[-5:]) > 220:
                break
        if plot:
            episode_reward_plot(all_rewards, self.steps_completed, window_size=7, step_size=1)
        return all_rewards, all_rewards_eval
    

    def choose_action(self, s):
        with torch.no_grad():
            state = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
            action = self.Actor(state)
        return action.cpu().numpy().flatten()


    def compute_critic_loss(self, batch):
        obs, action, reward, next_obs, terminated = batch

        with torch.no_grad():
            next_action = self.Actor_target(next_obs)

            noise = torch.normal(0, self.policy_noise, size=next_action.shape).to(device)
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_action = torch.clamp(next_action + noise, -1, 1)

            target_q1 = self.Critic1_target(next_obs, next_action)
            target_q2 = self.Critic2_target(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2)
            
            target_q = reward + (1 - terminated) * self.gamma * target_q

        current_q1 = self.Critic1(obs, action)
        current_q2 = self.Critic2(obs, action)

        loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        return loss
    

    def compute_actor_loss(self, batch):
        obs, _, _, _, _ = batch
        actor_loss = -self.Critic1(obs, self.Actor(obs)).mean()
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
    env = gym.make("LunarLander-v3", continuous=True, render_mode='rgb_array')

    td3_agent = TD3(env, replay_size=1000000, batch_size=100, gamma=0.99)

    td3_agent.learn(500000)
    env = RecordVideo(gym.make("LunarLander-v3", continuous=True, render_mode='rgb_array'), 'video_td3')    
    video_agent(env, td3_agent, n_episodes=5)  
    pass
