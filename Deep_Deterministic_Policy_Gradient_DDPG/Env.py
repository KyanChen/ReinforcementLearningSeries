import gym
# 'CartPole-v0'


class Env:
    def __init__(self, env):
        self.env = gym.make(env)
        self.env.unwrapped
        '''
        行为 Action 有三个，向左 (0)，向右 (2)，无(1) 推车。
        奖励: 除了超过目的地 (位置为 0.5)， 其余地方的奖励均为 "-1"
        '''
        print('action_space: ' + repr(self.env.action_space))
        print('observation_space: ' + repr(self.env.observation_space))
        print('observation_space.low: ' + repr(self.env.observation_space.low))
        print('observation_space.high: ' + repr(self.env.observation_space.high))
        self.n_features = self.env.observation_space.shape[0]

        self.action_bounds = self.env.action_space.high

    def start(self):
        return self.env.reset()

    def show(self):
        self.env.render()

    def step(self, a):
        return self.env.step(a)


if __name__ == '__main__':
    env = Env()

