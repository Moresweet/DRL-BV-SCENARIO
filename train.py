import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import FloatTensor, LongTensor
from collections import namedtuple
import random
import gymnasium as gym
import highway_env
from matplotlib import pyplot as plt
import numpy as np
import itertools

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

env = gym.make("bv-scenario-v0", render_mode='rgb_array')

# Actor-Critic网络模型基本参数
Tensor = FloatTensor
GAMMA = 0.99
MEMORY_CAPACITY = 10000
BATCH_SIZE = 256
LR = 0.001
time_step = 0.125
k = 1  # 参数k
mu = 0.8  # 摩擦系数
g = 9.81  # 重力加速度


class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.linear1 = nn.Linear(9, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 125)

    def forward(self, s):
        s = torch.FloatTensor(s)
        s = self.linear1(s)
        s = torch.relu(s)
        s = self.linear2(s)
        s = torch.relu(s)
        s = self.linear3(s)
        s = torch.relu(s)
        s = self.linear4(s)
        return torch.softmax(s, dim=-1)


class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.linear1 = nn.Linear(9, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 3)

    def forward(self, s):
        s = torch.FloatTensor(s)
        s = self.linear1(s)
        s = torch.relu(s)
        s = self.linear2(s)
        s = torch.relu(s)
        s = self.linear3(s)
        s = torch.relu(s)
        s = self.linear4(s)
        return s


class ActorCritic(object):
    def __init__(self):
        self.actor_net = ActorNet()
        self.critic_net1 = CriticNet()
        self.critic_net2 = CriticNet()
        self.target_critic_net1 = CriticNet()
        self.target_critic_net2 = CriticNet()
        self.optimizer_actor = torch.optim.Adam(self.actor_net.parameters(), lr=LR * 10)
        self.optimizer_critic1 = torch.optim.Adam(self.critic_net1.parameters(), lr=LR)
        self.optimizer_critic2 = torch.optim.Adam(self.critic_net2.parameters(), lr=LR)
        self.memory = []
        self.position = 0
        self.capacity = MEMORY_CAPACITY
        self.loss_func = nn.MSELoss()
        self.lambda_ = torch.tensor(0.1, requires_grad=True)  # 初始化对偶变量

    # def choose_action(self, s):
    #     s = torch.FloatTensor(s)  # 确保输入形状是(batch_size, input_dim)
    #     prob = self.actor_net(s).detach().numpy()
    #     # 仅保留合法动作的概率
    #     valid_actions = [0, 1, 2, 3, 4]
    #     valid_prob = np.array([prob[0][i] if i in valid_actions else 0 for i in range(len(prob[0]))])
    #
    #     # 归一化概率，使其和为1
    #     if np.all(np.isnan(valid_prob)) or np.sum(valid_prob) == 0:
    #         valid_prob = np.full_like(valid_prob, 1.0 / len(valid_actions))  # 将每个动作的概率均设置为0.2
    #     else:
    #         valid_prob /= valid_prob.sum()
    #
    #     # 确保动作合法
    #     action = np.random.choice(valid_actions, p=valid_prob)
    #     return action

    def choose_action(self, s):
        s = torch.FloatTensor(s)  # 确保输入形状是(batch_size, input_dim)
        prob = self.actor_net(s).detach().numpy()

        # 生成所有可能的动作组合
        valid_actions = list(itertools.product(range(5), repeat=3))

        # 计算每个组合的概率
        valid_prob = np.array([prob[0][i] * prob[0][j] * prob[0][k] for (i, j, k) in valid_actions])

        # 归一化概率，使其和为1
        if np.all(np.isnan(valid_prob)) or np.sum(valid_prob) == 0:
            valid_prob = np.full_like(valid_prob, 1.0 / len(valid_actions))  # 将每个动作的概率均设置为均等
        else:
            valid_prob /= valid_prob.sum()

        # 确保动作合法
        action_index = np.random.choice(len(valid_actions), p=valid_prob)
        action = valid_actions[action_index]

        return action
    def push_memory(self, s, a, r, s_):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(torch.unsqueeze(torch.FloatTensor(s), 0),
                                                torch.unsqueeze(torch.FloatTensor(s_), 0),
                                                torch.from_numpy(np.array([a])),
                                                torch.from_numpy(np.array([r], dtype='float32')))
        self.position = (self.position + 1) % self.capacity

    def get_sample(self, batch_size):
        sample = random.sample(self.memory, batch_size)
        return sample

    def learn(self):
        transitions = self.get_sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        b_s = Variable(torch.cat(batch.state))
        b_s_ = Variable(torch.cat(batch.next_state))
        b_a = Variable(torch.cat(batch.action))
        b_a = b_a.unsqueeze(1)
        b_r = Variable(torch.cat(batch.reward))
        b_r = b_r.unsqueeze(1)

        # Critic更新
        with torch.no_grad():
            q_target1 = b_r + GAMMA * self.target_critic_net1(b_s_).squeeze(1)
            q_target2 = b_r + GAMMA * self.target_critic_net2(b_s_).squeeze(1)
            q_target = torch.min(q_target1, q_target2)

        q_eval1 = self.critic_net1(b_s).squeeze(1).gather(0, b_a.squeeze(1).to(torch.int64))
        q_eval2 = self.critic_net2(b_s).squeeze(1).gather(0, b_a.squeeze(1).to(torch.int64))
        critic_loss1 = self.loss_func(q_eval1, q_target)
        critic_loss2 = self.loss_func(q_eval2, q_target)

        self.optimizer_critic1.zero_grad()
        critic_loss1.backward()
        self.optimizer_critic1.step()

        self.optimizer_critic2.zero_grad()
        critic_loss2.backward()
        self.optimizer_critic2.step()

        # Actor更新
        actions_prob = self.actor_net(b_s)
        b_a_index = b_a[..., 0] * 25 + b_a[..., 1] * 5 + b_a[..., 2]
        # chosen_action_prob = actions_prob.gather(2, b_a.to(torch.int64).unsqueeze(2))
        chosen_action_prob = torch.gather(actions_prob, 2, b_a_index.unsqueeze(2))
        advantage = (q_target - q_eval1).detach()
        actor_loss = -(torch.log(chosen_action_prob) * advantage * self.lambda_).mean()  # 使用对偶变量

        # 加入均衡动作选择的正则化项
        action_entropy = -torch.sum(actions_prob * torch.log(actions_prob), dim=1).mean()
        actor_loss -= 0.01 * action_entropy

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # Dual variable update
        self.lambda_ = self.lambda_ + 0.1 * (torch.mean(q_target - q_eval1).detach() - 0.01)

        # Target network update
        self.target_critic_net1.load_state_dict({name: 0.99 * param + 0.01 * target_param
                                                 for name, param, target_param in
                                                 zip(self.target_critic_net1.state_dict().keys(),
                                                     self.critic_net1.state_dict().values(),
                                                     self.target_critic_net1.state_dict().values())})
        self.target_critic_net2.load_state_dict({name: 0.99 * param + 0.01 * target_param
                                                 for name, param, target_param in
                                                 zip(self.target_critic_net2.state_dict().keys(),
                                                     self.critic_net2.state_dict().values(),
                                                     self.target_critic_net2.state_dict().values())})

        return actor_loss, critic_loss1, critic_loss2


Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward'))

# 训练模型
ac = ActorCritic()
count = 0

sum_reward = 0
all_reward = []

all_actor_loss = []
all_critic_loss1 = []
all_critic_loss2 = []

reward_threshold = 66  # 奖励上限


def get_distance_to_front_vehicle(observation):
    # 本车位置
    ego_vehicle = observation[0]
    ego_x = ego_vehicle[1]  # 获取x坐标
    ego_y = ego_vehicle[2]  # 获取y坐标

    # 初始化最小距离为一个较大的数
    min_distance = 999

    # 遍历其他车辆
    for vehicle in observation[1:]:
        vehicle_x = vehicle[1]  # 获取x坐标
        vehicle_y = vehicle[2]  # 获取y坐标

        # 仅考虑同车道并在前方的车辆
        if vehicle_x > ego_x and abs(vehicle_y - ego_y) < 2:  # 假设车道宽度为4，调整为适当的阈值
            distance = vehicle_x - ego_x
            if distance < min_distance:
                min_distance = distance

    return min_distance


def get_angular_velocity(current_heading, previous_heading, time_step):
    if previous_heading is None:
        return 0
    cos_h_current, sin_h_current = current_heading
    cos_h_previous, sin_h_previous = previous_heading
    heading_current = np.arctan2(sin_h_current, cos_h_current)
    heading_previous = np.arctan2(sin_h_previous, cos_h_previous)
    angular_velocity = (heading_current - heading_previous) / time_step
    return angular_velocity

for episode in range(2000):
    done = False
    s = env.reset()[0]
    s = torch.FloatTensor(s)
    reward = []
    episode_reward = 0  # 初始化累积奖励

    while not done:
        a = ac.choose_action(s)
        s_, r, done, info, _ = env.step(a)
        s_ = torch.FloatTensor(s_)
        env.render()
        # r = rewardf(s, r, a, info)  # Calculate reward based on next state s_ and action a
        ac.push_memory(s, a, r, s_)  # Store the transition in memory

        if len(ac.memory) >= BATCH_SIZE:
            actor_loss, critic_loss1, critic_loss2 = ac.learn()
            all_actor_loss.append(actor_loss)
            all_critic_loss1.append(critic_loss1)
            all_critic_loss2.append(critic_loss2)
            count += 1
            print('trained times:', count, r)
        # previous_heading = current_heading
        s = s_
        reward.append(r)
        episode_reward += r  # Accumulate the reward

        if episode_reward > reward_threshold:  # Check if reward threshold is reached
            break
    sum_reward = np.sum(reward)
    all_reward.append(sum_reward)

plt.plot(all_reward, 'b*--', alpha=0.5, linewidth=1, label='acc')
plt.show()

a1 = torch.Tensor(all_actor_loss)
plt.plot(a1.detach().numpy(), 'b*--', alpha=0.5, linewidth=1, label='actor_loss')
plt.show()

a2 = torch.Tensor(all_critic_loss1)
plt.plot(a2.detach().numpy(), 'r*--', alpha=0.5, linewidth=1, label='critic_loss1')
plt.show()

a3 = torch.Tensor(all_critic_loss2)
plt.plot(a3.detach().numpy(), 'g*--', alpha=0.5, linewidth=1, label='critic_loss2')
plt.show()
