import torch.nn as nn


class Actor(nn.Module):
    # action_bounds, 每个action的范围，默认为对称结构，保存形式为tensor:[2,2,2]
    def __init__(self, n_features, action_bounds):
        super(Actor, self).__init__()
        self.n_features = n_features
        self.action_bounds = action_bounds
        self.layers = nn.Sequential(
            nn.Linear(self.n_features, self.n_features * 8),
            nn.PReLU(),
            nn.Linear(self.n_features * 8, self.n_features * 8),
            nn.PReLU(),
            nn.Linear(self.n_features * 8, len(self.action_bounds)),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.layers(x)
        # Scale output to -action_bound to action_bound
        scaled_a = x * self.action_bounds
        return scaled_a

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

