import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datetime
from pathlib import Path
from info import HyperParameters, Params

class Actor3Critic(nn.Module):
    def __init__(self, state_dim, action_dim, device="cuda", stacked_frames=4, is_tuning=False):
        super().__init__()
        self.device = device
        self.is_tuning = is_tuning
        
        self.conv1 = nn.Conv2d(stacked_frames, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self._get_output((stacked_frames, 84, 84))
        
        self.actor_net = nn.Sequential(
            nn.Linear(self.output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )
        
        self.critic_net = nn.Sequential(
            nn.Linear(self.output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        if not is_tuning:
            self.critic_optim = optim.RMSprop(self.critic_net.parameters(), lr=HyperParameters["CRITIC_LR"])
            self.actor_optim = optim.RMSprop(self.actor_net.parameters(), lr=HyperParameters["ACTOR_LR"])
        else:
            self.critic_optim = optim.RMSprop(self.critic_net.parameters(), lr=Params["critic_lr"])
            self.actor_optim = optim.RMSprop(self.actor_net.parameters(), lr=Params["actor_lr"])
    
    def _get_output(self, shape):
        x = torch.zeros(1, *shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        self.output_size = int(x.nelement() / x.shape[0])
    
    def forward(self, x):
        return NotImplementedError("Use get_action or get_value")
    
    def _forward_conv(self, x, is_test):
        if type(x) == torch.Tensor:
            x = x.to(dtype=torch.float32, device="cpu")
        else:
            x = torch.tensor(x, dtype=torch.float32, device="cpu")
        if is_test:
            x = x.expand(1, 84, 84, 4)
            x = x.reshape(1, 4, 84, 84).to("cpu")
        else:
            x = x.reshape(HyperParameters["N_ENVS"], 4, 84, 84).to("cpu")
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        return x
    
    def get_action(self, x, is_test):
        x =  self._forward_conv(x, is_test)
        logits = self.actor_net(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy
    
    def get_value(self, x, is_test):
        x = self._forward_conv(x, is_test)
        return self.critic_net(x)
    
    def update_nets(self, critic_loss, actor_loss):
        if not self.is_tuning:
            grad_norm = HyperParameters["GRAD_NORM"]
        else:
            grad_norm = Params["grad_norm"]
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), grad_norm)
        self.critic_optim.step()
    
        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), grad_norm)
        self.actor_optim.step()
    
    def save_model(self, agent):
        save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        save_dir.mkdir(parents=True)
        actor_path = (save_dir / f"actor.chkpt")
        critic_path = (save_dir / f"critic.chkpt")
        actor_optim_path = (save_dir / f"aOptim.chkpt")
        critic_optim_path = (save_dir / f"cOptim.chkpt")
        torch.save(self.actor_net.state_dict(), actor_path)
        torch.save(agent.actor_optim.state_dict(), actor_optim_path)
        torch.save(agent.critic_net.state_dict(), critic_path)
        torch.save(agent.critic_optim.state_dict(), critic_optim_path)
    
    def load_weights(self, is_training=True):
        self.actor_net.load_state_dict(torch.load(r"./checkpoints/2024-07-01T08-14-24/actor.chkpt"))
        self.actor_optim.load_state_dict(torch.load(r"./checkpoints/2024-07-01T08-14-24/aOptim.chkpt"))
        self.critic_net.load_state_dict(torch.load(r"./checkpoints/2024-07-01T08-14-24/critic.chkpt"))
        self.critic_optim.load_state_dict(torch.load(r"./checkpoints/2024-07-01T08-14-24/cOptim.chkpt"))
        if not is_training:
            self.actor_net.eval()
            self.critic_net.eval()
        else:
            self.actor_net.train()
            self.critic_net.train()
        