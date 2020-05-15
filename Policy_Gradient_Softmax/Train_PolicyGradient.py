import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import matplotlib.pyplot as plt

import Config
import Env
from PolicyNetwork import PolicyNetwork


class PolicyGradient:
    def __init__(self):
        self.policy_net = PolicyNetwork()
        self.n_actions = Config.N_ACTIONS
        self.states, self.actions, self.rewards = [], [], []
        self.gamma = Config.REWARD_DECAY
        self.load_weights(self.policy_net)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), Config.LR, (0.9, 0.99))

    def load_weights(self, net):
        # net.state_dict(), 得出来的名字，'layers.1.weight'
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 1)
                nn.init.constant_(m.bias, 0)

    def store_trajectory(self, s, a, r):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)

    def chose_action(self, s):
        s = torch.Tensor(np.expand_dims(s, axis=0))
        # 每个动作的概率
        actions_probs = F.softmax(self.policy_net(s).detach(), dim=1)

        # 根据概率选动作
        action = random.choices(range(self.n_actions), weights=actions_probs.squeeze(0))[0]
        return action

    def learn(self):
        # discount and normalize episode reward
        discount_and_norm_rewards = torch.from_numpy(self._discount_and_norm_rewards()).float()
        states = torch.from_numpy(np.concatenate(self.states, axis=0)).float()
        actions = torch.from_numpy(np.concatenate(self.actions, axis=0)).long()
        log_probs = nn.NLLLoss(reduction='none')(nn.LogSoftmax(dim=1)(self.policy_net(states)), actions)
        loss = torch.sum(log_probs * discount_and_norm_rewards)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discount_and_norm_rewards = np.array([])
        for episode_rewards in self.rewards:
            discounted_episode_rewards = np.zeros_like(episode_rewards)
            running_reward = 0
            for t in reversed(range(0, len(episode_rewards))):
                running_reward = self.gamma * running_reward + episode_rewards[t]
                discounted_episode_rewards[t] = running_reward
            # normalize episode rewards
            discounted_episode_rewards -= np.mean(discounted_episode_rewards)
            discounted_episode_rewards /= np.std(discounted_episode_rewards)
            discount_and_norm_rewards = np.concatenate(
                (discount_and_norm_rewards, discounted_episode_rewards), axis=0)
        return discount_and_norm_rewards

    def draw_curve(self, loss):
        x = np.arange(1, len(loss)+1)
        plt.title("cost curve")
        plt.xlabel("train step")
        plt.ylabel("cost")
        plt.plot(x, loss)
        plt.show()


if __name__ == '__main__':
    policy_gradient = PolicyGradient()
    env = Env.Env()
    for i_epoch in range(10000):
        for i_episode in range(Config.EPISODES_LEARN_ONCE):
            s = env.start()
            old_a = None

            states = []
            actions = []
            rewards = []

            for _ in range(500):
                env.show()
                a = policy_gradient.chose_action(s)
                current_a = a
                s_, r, done, info = env.step(a)
                position, velocity = s_
                if r == 1:
                    r = 100
                else:
                    r = math.log(position + 1.3)
                if (old_a, current_a) in [(0, 2), (2, 0)]:
                    r += -5
                if current_a == 1:
                    r += -1
                # 将轨迹添加
                states.append(s)
                actions.append(a)
                rewards.append(r)
                # 如果该轨迹完成了，则保存轨迹
                if done:
                    policy_gradient.store_trajectory(states, actions, rewards)
                    break
                s = s_
                old_a = current_a

            print("episode:", i_epoch*Config.EPISODES_LEARN_ONCE + i_episode, "  reward:", int(sum(rewards)))
        loss = policy_gradient.learn()


