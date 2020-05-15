import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, n_features, actions, is_continues=None):
        super(Actor, self).__init__()
        self.n_features = n_features
        self.n_actions = actions
        self.is_continues = is_continues
        '''
        如果是连续动作空间
        action：从当前状态s,通过演员动作网络，预测正态分布的u值，
        从当前状态s,通过演员动作网络，预测正态分布的sigma值，
        通过u、sigma抽样1个样本 构建action
        根据Mu和sigma求出一个正太分布，这个是随机的正态分布
        '''
        self.layers = nn.Sequential(
            nn.Linear(self.n_features, self.n_features * 8),
            nn.PReLU(),
            nn.Linear(self.n_features * 8, self.n_features * 8),
            nn.PReLU()
        )
        if self.is_continues:
            self.mu = nn.Sequential(
                nn.Linear(self.n_features * 8, 1),
                nn.Tanh())
            self.sigma = nn.Sequential(
                nn.Linear(self.n_features * 8, 1),
                nn.Softplus()
            )
        else:
            self.action_prb = nn.Linear(self.n_features * 8, self.n_actions)

    def forward(self, x):
        x = self.layers(x)
        if self.is_continues:
            return self.mu(x), self.sigma(x)
        else:
            return self.action_prb(x)

    '''
    # 迭代循环初始化参数
    for m in self.children():
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, -100)
        # 也可以判断是否为conv2d，使用相应的初始化方式 
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.item(), 1)
            nn.init.constant_(m.bias.item(), 0)   
    '''

