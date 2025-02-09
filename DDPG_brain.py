import torch
import torch.nn.functional as F
import numpy as np

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1) # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)

class DDPG:
    ''' DDPG算法 '''

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)  # 目标策略网络
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)  # 目标价值网络
        self.target_actor.load_state_dict(self.actor.state_dict())  # 初始化目标策略网络并设置和策略相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())  # 初始化目标价值网络并设置和价值网络相同的参数
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma  # 贪心系数
        self.sigma = sigma  # 高斯噪声的标准差，均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

    def take_action(self, state):  # 给动作添加噪声，增加探索
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        # self.actor.eval()
        action = self.actor(state).item()
        # self.actor.train()
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action

    def soft_update(self, net, target_net):  # 软更新函数
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)  # TD目标
        # 训练价值网络
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))  # 均方误差损失函数 更新价值网络的参数 梯度下降
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # 训练策略网络
        actor_loss = -torch.mean(self.critic(states, self.actor(states)))  # 均方误差损失函数 更新策略网络的参数 梯度上升
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # 让目标网络缓慢更新，逐渐接近原网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络
        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
