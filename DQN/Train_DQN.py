import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt

import Config
import Env
from QNet import DQNNet


class DQN:
    def __init__(self):
        self.memory = []
        self.memory_size = Config.MEMORY_SIZE
        self.memory_counter = 0
        self.q_eval = DQNNet()
        self.load_weights(self.q_eval)
        self.q_target = DQNNet()
        self.n_actions = Config.N_ACTIONS
        self.learn_iter = 0
        self.replace_target_net_iter = Config.REPLACE_TARGET_NET_ITER
        self.batch_size = Config.BATCH_SIZE
        self.gamma = Config.REWARD_DECAY
        self.optimizer = torch.optim.Adam(self.q_eval.parameters(), Config.LR, (0.9, 0.99))
        self.loss_curve = []

    def load_weights(self, net):
        # net.state_dict(), 得出来的名字，'layers.1.weight'
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 1)
                nn.init.constant_(m.bias, 0)

    def store_transition(self, s, a, r, s_):
        s, a, r, s_ = torch.FloatTensor(s), torch.Tensor([a]), torch.FloatTensor([r]), torch.FloatTensor(s_)
        if self.memory_counter < self.memory_size:
            self.memory.append((s, a, r, s_))
        else:
            index = self.memory_counter % self.memory_size
            self.memory[index] = (s, a, r, s_)
        self.memory_counter += 1

    def chose_action(self, s, e_greedy):
        if np.random.uniform() < e_greedy:
            s = torch.Tensor(np.expand_dims(s, axis=0))
            actions_q_value = self.q_eval.forward(s)
            action = int(torch.argmax(actions_q_value, dim=1)[0].detach().cpu().numpy())
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_iter % self.replace_target_net_iter == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())
        self.learn_iter += 1
        samples = random.sample(self.memory, k=self.batch_size)
        s = [i[0] for i in samples]
        s = torch.cat(s, dim=0).view(Config.BATCH_SIZE, -1)
        a = [i[1] for i in samples]
        a = torch.cat(a, dim=0).view(Config.BATCH_SIZE, -1)
        r = [i[2] for i in samples]
        r = torch.cat(r, dim=0).view(Config.BATCH_SIZE, -1)
        s_ = [i[3] for i in samples]
        s_ = torch.cat(s_, dim=0).view(Config.BATCH_SIZE, -1)
        q_s_a = self.q_eval(s).gather(1, a.long())
        q_s_ = torch.max(self.q_target(s_).detach(), dim=1)[0].unsqueeze(dim=1)
        q_target = r + self.gamma * q_s_

        loss = nn.MSELoss()(q_s_a, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_curve.append(loss)

    def draw_curve(self):
        x = np.arange(1, len(self.loss_curve)+1)
        plt.title("cost curve")
        plt.xlabel("train step")
        plt.ylabel("cost")
        plt.plot(x, self.loss_curve)
        plt.show()


total_step = 0
if __name__ == '__main__':
    dqn = DQN()
    env = Env.Env()
    for i_episode in range(300):
        e_greedy = Config.E_GREEDY + i_episode * (1-Config.E_GREEDY) / 100
        print(i_episode)
        s = env.start()
        while True:
            env.show()
            a = dqn.chose_action(s, e_greedy)
            s_, r, done, info = env.step(a)
            position, velocity = s_
            r = (position + 0.5) ** 2
            if r == 1:
                r = 1000
            if done:
                print(s_, r, done, info)
                break
            else:
                dqn.store_transition(s, a, r, s_)
            if total_step > Config.MEMORY_SIZE:
                dqn.learn()
            s = s_
            total_step += 1
    dqn.draw_curve()

