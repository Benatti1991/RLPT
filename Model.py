import torch
import torch.nn as nn
from torch.distributions import Normal
import copy

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        #self.log_std = self.log_std.clamp(-20, -1)

        self.apply(init_weights)

    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value


class ActorCriticPCoady(ActorCritic):
    def __init__(self, num_inputs, num_outputs, std=-2.3):
        nn.Module.__init__(self)
        h1 = num_inputs * 10
        h3 = num_outputs * 10
        h2 = int(pow(h1*h3,0.5))
        h2_cr = int(pow(h1*10,0.5))
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, h1),
            nn.Tanh(),
            nn.Linear(h1, h2_cr),
            nn.Tanh(),
            nn.Linear(h2_cr, 10),
            nn.Tanh(),
            nn.Linear(10, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
            nn.Linear(h2, h3),
            nn.Tanh(),
            nn.Linear(h3, num_outputs),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

        self.apply(init_weights)

# From baselines Policies.py , line 15
def outputSize(in_size, kernel_size, stride, padding):
    conv_size = copy.deepcopy(in_size)
    for i in range(len(kernel_size)):
        conv_size[0] = int((conv_size[0] - kernel_size[i] + 2*(padding[i])) / stride[i]) + 1
        conv_size[1] = int((conv_size[1] - kernel_size[i] + 2*(padding[i])) / stride[i]) + 1

    return(conv_size)

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class ActorCritic_nature_cnn(ActorCritic):
    # CNN from Nature paper.
    def __init__(self, image_shape, num_outputs, std=-2.3):
        super(ActorCritic, self).__init__()
        self.input_shape = image_shape
        fc_size = outputSize(image_shape, [8,4,3], [4,2,1], [0,0,0])
        self.actor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            Flatten(),
            nn.Linear(fc_size[0] * fc_size[1] * 64, num_outputs),
            nn.ReLU()
        )

        self.critic = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            Flatten(),
            nn.ReLU(),
            nn.Linear(fc_size[0] * fc_size[1] * 64, 1)
        )

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

        self.apply(init_weights)


    def forward(self, x):
        # Input shape is [batch, Height, Width, RGB] Torch wants [batch, RGB, Height, Width]. Must Permute
        x = (x.permute(0,3,1,2)/255.0)
        value = self.critic(x)
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value


class MultiSensorSimple(nn.Module):
    def __init__(self, image_shape, sens2_shape, num_outputs, std=-2.3):
        super(MultiSensorSimple, self).__init__()
        self.input_shape = image_shape
        fc_size = outputSize(image_shape, [8, 4, 3], [4, 2, 1], [0, 0, 0])
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            Flatten(),
            nn.ReLU(),
            nn.Linear(fc_size[0] * fc_size[1] * 64, num_outputs * 5),

        )
        self.actor_fc0 = nn.Linear(sens2_shape, num_outputs * 5)
        self.actor_fc1 = nn.Linear(num_outputs * 10, num_outputs * 5)
        self.actor_fc2 = nn.Linear(num_outputs * 5, num_outputs)

        self.critic_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            Flatten(),
            nn.ReLU(),
            nn.Linear(fc_size[0] * fc_size[1] * 64, num_outputs * 5),

        )
        self.critic_fc0 = nn.Linear(sens2_shape, num_outputs * 5)
        self.critic_fc1 = nn.Linear(num_outputs * 10, num_outputs * 5)
        self.critic_fc2 = nn.Linear(num_outputs * 5, 1)


        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        self.apply(init_weights)


    def forward(self, data):
        x0 = (data[0].permute(0, 3, 1, 2) / 255)
        x1 = self.actor_cnn(x0)#.view(-1)
        x2 = self.actor_fc0(data[1])
        x = torch.cat((x1, x2), dim=1)
        x = nn.functional.relu(self.actor_fc1(x))
        x = self.actor_fc2(x)
        mu = torch.tanh(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)

        y1 = self.critic_cnn(x0)#.view(-1)
        y2 = self.critic_fc0(data[1])
        y = torch.cat((y1, y2), dim=1)
        y = nn.functional.relu(self.critic_fc1(y))
        value = self.critic_fc2(y)

        return dist, value