import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import matplotlib.pyplot as plt

import Config
import Env
from Actor import Actor
from Critic import Critic


class Actor_Critic:
    def __init__(self, n_features, actions=None, is_continues=None):
        self.actions = actions
        self.is_continues = is_continues
        self.actor_net = Actor(n_features, actions=actions, is_continues=is_continues)
        self.critic_net = Critic(n_features)
        self.load_weights(self.actor_net)
        self.load_weights(self.critic_net)
        # we need a good teacher, so the teacher should learn faster than the actor
        self.optimizer_actor = torch.optim.Adam(self.actor_net.parameters(), Config.LR_ACTOR, (0.9, 0.99))
        self.optimizer_critic = torch.optim.Adam(self.critic_net.parameters(), Config.LR_CRITIC, (0.9, 0.99))
        self.gamma = Config.REWARD_DECAY

    def load_weights(self, net):
        # net.state_dict(), 得出来的名字，'layers.1.weight'
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 1)
                nn.init.constant_(m.bias, 0.1)

    def store_trajectory(self, s, a, r):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)

    def chose_action(self, s):
        s = torch.Tensor(np.expand_dims(s, axis=0))
        if self.is_continues:
            mu, sigma = self.actor_net(s)
            mu, sigma = mu.detach().squeeze(), sigma.detach().squeeze()
            normal_dist = torch.distributions.Normal(mu*2, sigma+0.1)
            action = torch.clamp(normal_dist.sample((1,)), min=-self.actions[0], max=self.actions[0])
        else:
            # 每个动作的概率
            actions_probs = F.softmax(self.actor_net(s).detach(), dim=1)

            # 根据概率选动作
            action = random.choices(range(actions_probs.size(1)), weights=actions_probs.squeeze(0))[0]
        return action

    def learn(self, s, a, r, s_):
        s = torch.from_numpy(s).unsqueeze(dim=0).float()
        s_ = torch.from_numpy(s_).unsqueeze(dim=0).float()
        r = torch.tensor(r)
        a = torch.tensor(a).unsqueeze(dim=0)
        V_st = self.critic_net(s).squeeze(dim=0)
        V_st_ = self.critic_net(s_).squeeze(dim=0)
        # td_error = Q(st, at) - V(st)
        # Q(st, at) = r + r*V(st+1)
        # td_error = - V(st) + r + gamma*V(st+1)
        td_error = r + self.gamma * V_st_ - V_st
        loss_critic = td_error ** 2

        # critic 学习过程
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        # actor 学习过程
        mu, sigma = self.actor_net(s)
        mu, sigma = mu.squeeze(), sigma.squeeze()
        normal_dist = torch.distributions.Normal(mu*2, sigma+0.1)
        log_prob = normal_dist.log_prob(a)
        loss_actor = torch.sum(log_prob * td_error.detach())
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()
        return loss_critic, loss_actor

    def draw_curve(self, loss):
        x = np.arange(1, len(loss)+1)
        plt.title("cost curve")
        plt.xlabel("train step")
        plt.ylabel("cost")
        plt.plot(x, loss)
        plt.show()


if __name__ == '__main__':
    env_name = 'Pendulum-v0'
    env = Env.Env(env_name)
    actor_critic = Actor_Critic(env.n_features, actions=env.actions, is_continues=env.is_continues)

    for i_epoch in range(10000):
        for i_episode in range(Config.MAX_EPISODE):
            s = env.start()
            old_a = None
            rewards = []

            while True:
                env.show()
                a = actor_critic.chose_action(s)
                current_a = a
                s_, r, done, info = env.step(a)
                # 将Reward添加
                rewards.append(r)
                actor_critic.learn(s, a, r, s_)
                s = s_
                old_a = current_a
                if done:
                    print("episode:", i_epoch*Config.MAX_EPISODE + i_episode, "  reward:", sum(rewards))
                    break


