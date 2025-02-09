import random
import numpy as np
import gym
import matplotlib.pyplot as plt
import rl_utils
from DDPG_brain import DDPG
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

actor_lr = 3e-4  # 梯度下降 演员 学习率
critic_lr = 3e-3  # 梯度下降 批判家 学习率
num_episodes = 150  # 回合数
hidden_dim = 64  # 隐藏层神经元个数
gamma = 0.9  # 贪心系数
tau = 0.005  # 软更新参数
buffer_size = 10000  # 经验池容量
minimal_size = 1000  # 经验池超过minimal_size后再训练
batch_size = 64  # 一次所取得的经验池数据个数
sigma = 0.01  # 高斯噪声标准差
# GPU运算
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# 加载实验环境 倒立摆
env_name = 'Pendulum-v1'
env = gym.make("Pendulum-v1", render_mode="human")
# 加载实验环境 登山车
# env_name = 'MountainCarContinuous-v0'
# env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")

# 设置随机种子 通过设置种子，程序每次运行时生成的随机数序列都是相同的，这对于调试和实验复现非常重要。
random.seed(0)  # 设置Python标准库的随机种子
np.random.seed(0)  # 设置NumPy的随机种子
torch.manual_seed(0)  # 设置PyTorch的随机种子
state = env.reset(seed=0)  # 对于Gym >= 0.19创建Gym环境并设置随机种子
env.seed(0)  # 如果使用较旧版本的Gym，可以使用env.seed(0)

replay_buffer = rl_utils.ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]  # 动作最大值
agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)
return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)


# DDPG训练
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format(env_name))
plt.minorticks_on()
plt.grid(which='minor', linestyle='--', linewidth=0.75)
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format(env_name))
plt.minorticks_on()
plt.grid(which='minor', linestyle='--', linewidth=0.75)
plt.show()