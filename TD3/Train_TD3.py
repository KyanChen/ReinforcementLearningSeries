import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import matplotlib.pyplot as plt
import copy

import Config
import Env
from Actor import Actor
from Critic import Critic


class TD3:
    def __init__(self, n_features, action_bounds):
        self.n_features = n_features
        self.action_bounds = action_bounds

        self.eval_actor_net = Actor(n_features, action_bounds)
        self.load_weights(self.eval_actor_net)
        self.eval_actor_net.train()
        self.target_actor_net = copy.deepcopy(self.eval_actor_net)
        self.target_actor_net.eval()

        self.eval_critic_net1 = Critic(n_features, action_bounds)
        self.load_weights(self.eval_critic_net1)
        self.eval_critic_net1.train()

        self.eval_critic_net2 = Critic(n_features, action_bounds)
        self.load_weights(self.eval_critic_net2)
        self.eval_critic_net2.train()

        self.target_critic_net1 = copy.deepcopy(self.eval_critic_net1)
        self.target_critic_net1.eval()
        self.target_critic_net2 = copy.deepcopy(self.eval_critic_net2)
        self.target_critic_net2.eval()

        self.memory = Memory(Config.MEMORY_CAPACITY)
        self.batch_size = Config.BATCH_SIZE
        self.tau = Config.REPLACEMENT_SOFT_TAU

        # we need a good teacher, so the teacher should learn faster than the actor
        self.optimizer_actor = torch.optim.Adam(self.eval_actor_net.parameters(), Config.LR_ACTOR, (0.9, 0.99))
        # itertools.chain(self.encoder.parameters(), self.decoder.parameters())
        # self.optimizer_critic = \
        #     torch.optim.Adam([{'params': self.eval_critic_net1.parameters()},
        #                       {'params': self.eval_critic_net2.parameters()}], Config.LR_CRITIC, (0.9, 0.99))
        self.optimizer_critic1 = \
            torch.optim.Adam(self.eval_critic_net1.parameters(), Config.LR_CRITIC, (0.9, 0.99))
        self.optimizer_critic2 = \
            torch.optim.Adam(self.eval_critic_net2.parameters(), Config.LR_CRITIC, (0.9, 0.99))

        self.gamma = Config.REWARD_DECAY
        self.policy_noise_clip = Config.POLICY_NOISE_CLIP
        self.policy_delay = Config.DELAY_POLICY_UPDATE_ITER
        self.learn_iter = 0

    def load_weights(self, net):
        # net.state_dict(), 得出来的名字，'layers.1.weight'
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 1)
                nn.init.constant_(m.bias, 0.1)

    def store_transition(self, s, a, r, s_):
        self.memory.store([s, a, r, s_])

    def chose_action(self, s):
        s = torch.Tensor(np.expand_dims(s, axis=0))
        action = self.eval_actor_net(s).detach().squeeze(dim=0)
        return action

    def learn(self):
        self.learn_iter += 1
        # for x in self.Actor_target.state_dict().keys():
        #     eval('self.Actor_target.' + x + '.data.mul_((1-TAU))')
        #     eval('self.Actor_target.' + x + '.data.add_(TAU*self.Actor_eval.' + x + '.data)')
        # for x in self.Critic_target.state_dict().keys():
        #     eval('self.Critic_target.' + x + '.data.mul_((1-TAU))')
        #     eval('self.Critic_target.' + x + '.data.add_(TAU*self.Critic_eval.' + x + '.data)')

        # for target_param, param in zip(net_target.parameters(), net.parameters()):
        #     target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        # for k, v in self.eval_critic_net.state_dict().items():
        #     self.target_critic_net.state_dict()[k].copy_(self.tau * v + (1-self.tau) * self.target_critic_net.state_dict()[k])
        # for k, v in self.eval_actor_net.state_dict().items():
        #     self.target_actor_net.state_dict()[k].copy_(self.tau * v + (1-self.tau) * self.target_actor_net.state_dict()[k])

        batch_data = self.memory.sample(self.batch_size)
        s0, a0, r1, s1 = zip(*batch_data)
        s0 = torch.tensor(s0, dtype=torch.float)
        a0 = torch.tensor(a0, dtype=torch.float).view(self.batch_size, len(self.action_bounds))
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size, -1)
        s1 = torch.tensor(s1, dtype=torch.float)

        # Select action according to policy and add clipped noise

        # Input (s, a), output q
        q_s0_a0_1 = self.eval_critic_net1(s0, a0)
        q_s0_a0_2 = self.eval_critic_net2(s0, a0)
        # Input (s_, a_), output q_ for q_target
        # 得到a_
        noise = (torch.randn_like(a0) * self.policy_noise_clip * 2).clamp(
            -self.policy_noise_clip, self.policy_noise_clip)
        a1 = self.target_actor_net(s1).detach() + noise
        action_bound = self.action_bounds.expand_as(a1)
        a1[a1 < -action_bound] = - action_bound[a1 < -action_bound]
        a1[a1 > action_bound] = action_bound[a1 > action_bound]

        q_s1_a1_1 = self.target_critic_net1(s1, a1).detach()
        q_s1_a1_2 = self.target_critic_net2(s1, a1).detach()
        q_s1_a1 = torch.min(q_s1_a1_1, q_s1_a1_2)
        q_target = r1 + self.gamma * q_s1_a1

        loss_critic = nn.MSELoss()(q_s0_a0_1, q_target) + nn.MSELoss()(q_s0_a0_2, q_target)

        # critic 学习过程
        # # td_error=R + GAMMA * ct（bs_,at(bs_)）-ce(s,ba) 更新ce ,
        # 但这个ae(s)是记忆中的ba，让ce得出的Q靠近Q_target,让评价更准确
        # loss = (Q(st, at) - (rt + r*Q'(st+1, u'(st+1))))**2
        self.optimizer_critic1.zero_grad()
        self.optimizer_critic2.zero_grad()
        loss_critic.backward()
        self.optimizer_critic1.step()
        self.optimizer_critic2.step()
        loss_actor = 0
        # actor 学习过程
        # https://zhuanlan.zhihu.com/p/84321382
        # Delayed policy updates
        if self.learn_iter % self.policy_delay == 0:
            actor_a = self.eval_actor_net(s0)
            critic_q = self.eval_critic_net1(s0, actor_a)
            # loss=-q=-ce（s,ae（s））更新ae   ae（s）=a   ae（s_）=a_
            # 如果 a是一个正确的行为的话，那么它的Q应该更贴近0
            loss_actor = -torch.mean(critic_q)

            self.optimizer_actor.zero_grad()
            loss_actor.backward()
            self.optimizer_actor.step()
            # Update the frozen target models
            for param, target_param in zip(self.eval_critic_net1.parameters(), self.target_critic_net1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.eval_critic_net2.parameters(), self.target_critic_net2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.eval_actor_net.parameters(), self.target_actor_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return loss_critic, loss_actor

    def draw_curve(self, loss):
        x = np.arange(1, len(loss)+1)
        plt.title("cost curve")
        plt.xlabel("train step")
        plt.ylabel("cost")
        plt.plot(x, loss)
        plt.show()


class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.pointer = 0

    def store(self, transition):
        if len(self.memory) == self.capacity:
            self.memory.pop(0)
        self.memory.append(transition)
        self.pointer += 1

    def sample(self, batch_size):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=batch_size)
        return [self.memory[i] for i in indices]


if __name__ == '__main__':
    env_name = 'Pendulum-v0'
    env = Env.Env(env_name)
    action_bounds = torch.from_numpy(env.action_bounds)
    explore_var = action_bounds.detach().clone()
    td3 = TD3(env.n_features, action_bounds)

    for i_epoch in range(10000):
        for i_episode in range(Config.MAX_EPISODE):
            s = env.start()
            old_a = None
            episode_rewards = 0

            for i_iter in range(5000):
                env.show()
                if td3.memory.pointer < Config.MEMORY_CAPACITY:
                    a = env.sample_action()
                else:
                    a = td3.chose_action(s)
                    # 加入探索
                    for a_dim in range(len(a)):
                        a[a_dim] = torch.clamp(
                            torch.normal(a[a_dim], explore_var[a_dim]),
                            min=-action_bounds[a_dim], max=action_bounds[a_dim])
                current_a = a
                s_, r, done, info = env.step(a)

                # 存储到记忆池
                td3.store_transition(s, a, r, s_)
                # 如果记忆池满了开始学习
                if td3.memory.pointer > Config.MEMORY_CAPACITY:
                    # 减小探索度
                    explore_var *= 0.9999
                    explore_var = torch.clamp(explore_var, min=0.05)
                    td3.learn()
                s = s_
                episode_rewards += r
                old_a = current_a
                if done or i_iter > 1998:
                    print("episode:", i_epoch*Config.MAX_EPISODE + i_episode, "  reward:", episode_rewards)
                    break


