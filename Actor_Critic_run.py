import gym
import matplotlib.pyplot as plt
import rl_utils
from Actor_Critic_brain import ActorCritic
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 训练参数
actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
# GPU运算
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# 加载实验环境
env_name = 'CartPole-v1'
env = gym.make("CartPole-v1")

# , render_mode="human"
env.seed(0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device)
return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)


# Actor-Critic训练
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic on {}'.format(env_name))
plt.minorticks_on()
plt.grid(which='minor', linestyle='--', linewidth=0.75)
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic on {}'.format(env_name))
plt.minorticks_on()
plt.grid(which='minor', linestyle='--', linewidth=0.75)
plt.show()