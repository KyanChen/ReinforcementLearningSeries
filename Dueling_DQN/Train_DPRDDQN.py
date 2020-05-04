import numpy as np
import torch
import torch.nn as nn
import random
import math
import matplotlib.pyplot as plt

import Config
import Env
from QNet import DQNNet


# 用于四元组存储的类
class SumTree(object):
    def __init__(self, capacity):
        self.capacity = capacity
        # 存储优先级的tree结构和存储数据的data
        # tree结构我们使用一维数组实现，
        # 采取从上往下，从左往右的层次结构进行存储,
        # 同时，我们定义一个返回树根节点也就是树中叶子结点总优先级的函数
        self.data_pointer = 0
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    @property
    def total_p(self):
        return self.tree[0]  # the root

    # 定义一个用于添加数据的add函数，
    # 在添加数据的时候会触发我们的update函数，用于更新树中节点的值
    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)

        self.data_pointer += 1
        # replace when exceed the capacity
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    # 在添加数据的时候，由于某个叶子结点的数值改变了，
    # 那么它的一系列父节点的数值也会发生改变，
    # 所以定义一个update函数如下：
    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p

        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    # 定义一个根据数字来采样节点的算法，
    # 如何采样我们刚才已经介绍过了，
    # 即从头节点开始，每次决定往左还是往右，
    # 直到到达叶子结点为止，并返回叶子结点的id，
    # 优先级以对应的转移数据：
    def get_leaf(self, v):
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]


class Memory(object):
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.epsilon = 0.01  # small amount to avoid zero priority
        self.alpha = 0.6  # [0~1] convert the importance of TD error to priority
        self.beta = 0.4  # importance-sampling, from initial value increasing to 1
        self.beta_increment_per_sampling = 0.001
        self.abs_err_upper = 1.  # clipped abs error

    # 定义一个store函数，用于将新的经验数据存储到Sumtree中,
    # 我们定义了一个abs_err_upper和epsilon，
    # 表明p的范围在[epsilon,abs_err_upper]之间，
    # 对于第一条存储的数据，我们认为它的优先级P是最大的，
    # 同时，对于新来的数据，我们也认为它的优先级与当前树中优先级最大的经验相同。
    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p

    # 我们定义了一个采样函数,
    # 根据batch的大小对经验进行采样，
    # 采样的过程调用的是tree.get_leaf方法。
    # 同时在采样的过程中，
    # 我们还要计算在进行参数更新时每条数据的权重，
    # 代码之中权重的计算是对原文中的公式进行了修改
    # https: // www.jianshu.com / p / db14fdc67d2c
    def sample(self, n):
        b_idx, b_memory, ISWeights = \
            np.empty((n,), dtype=np.int32), \
            list(np.empty(n)), \
            np.empty((n, 1))

        pri_seg = self.tree.total_p / n

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        # for later calculate ISweight
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p

        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i] = idx, data
        return b_idx, b_memory, ISWeights

    # 更新树中权重的方法
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class DQN:
    def __init__(self):
        self.memory = Memory(Config.MEMORY_SIZE)

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
        # 在Q网络中，TD误差就是目标Q网络计算的目标Q值和当前Q网络计算的Q值之间的差距。
        s, a, r, s_ = torch.FloatTensor(s), torch.Tensor([a]), torch.FloatTensor([r]), torch.FloatTensor(s_)
        self.memory.store((s, a, r, s_))

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

        tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        s = [i[0] for i in batch_memory]
        s = torch.cat(s, dim=0).view(Config.BATCH_SIZE, -1)
        a = [i[1] for i in batch_memory]
        a = torch.cat(a, dim=0).view(Config.BATCH_SIZE, -1)
        r = [i[2] for i in batch_memory]
        r = torch.cat(r, dim=0).view(Config.BATCH_SIZE, -1)
        s_ = [i[3] for i in batch_memory]
        s_ = torch.cat(s_, dim=0).view(Config.BATCH_SIZE, -1)
        # q估计值
        q_eval = self.q_eval(s).gather(1, a.long())
        # s_在q_eval上选动作
        a_s_ = torch.argmax(self.q_eval(s_).detach(), dim=1)
        # 在q_target上得出来q值
        q_s__a = self.q_target(s_).gather(1, a_s_.unsqueeze(1).long()).detach()
        q_target = r + self.gamma * q_s__a

        loss = torch.sum(torch.tensor(ISWeights).t().float() @ ((q_eval - q_target) ** 2)) / self.batch_size
        abs_errors = abs(q_target - q_eval).view(-1).detach().cpu().numpy()
        self.memory.batch_update(tree_idx, abs_errors)
        # loss = nn.MSELoss()(q_eval, q_target)

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
    for i_episode in range(1000):
        e_greedy = Config.E_GREEDY + i_episode * (1-Config.E_GREEDY) / 100
        print(i_episode)
        s = env.start()
        current_a = 0
        for _ in range(500):
            env.show()
            old_a = current_a
            a = dqn.chose_action(s, e_greedy)
            current_a = a
            s_, r, done, info = env.step(a)
            position, velocity = s_
            r = abs(position - (-0.5))
            dqn.store_transition(s, a, r, s_)
            if total_step > Config.MEMORY_SIZE:
                dqn.learn()
            if done:
                break
                pass
            s = s_
            total_step += 1
    dqn.draw_curve()

