import torch.nn as nn
import torch
import Config


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.n_features = Config.N_FEATURES
        self.n_actions = Config.N_ACTIONS

        self.layers = nn.Sequential(
            nn.Linear(self.n_features, self.n_features * 8),
            nn.PReLU(),
            nn.Linear(self.n_features * 8, self.n_features * 8),
            nn.PReLU(),
            nn.Linear(self.n_features * 8, self.n_actions)
        )

    def forward(self, x):
        return self.layers(x)

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


if __name__ == '__main__':
    policy_network = PolicyNetwork()
    x = torch.Tensor((2, 3))
    output = policy_network.forward(x)
    print(output)

