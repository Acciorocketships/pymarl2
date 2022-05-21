import copy

import numpy as np

from pymarl.components.episode_buffer import EpisodeBatch
from pymarl.modules.critics.coma_gnn import ComaGNNCritic
from pymarl.utils.rl_utils import build_td_lambda_targets
from pymarl.components.transforms import OneHot
from torch.distributions import Categorical
import torch
from torch.optim import Adam

torch.autograd.set_detect_anomaly(True)


class ComaGNNLearner:
    def __init__(self, mac, scheme, logger, callback, args, **kwargs):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.critic = ComaGNNCritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.agent_params = list(mac.parameters())
        self.critic_params = list(self.critic.parameters())
        self.params = self.agent_params + self.critic_params

        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.critic_lr)

        self.onehot_transform = OneHot(self.n_actions)
        self.baseline_agg = lambda x, dim: torch.max(x, dim=dim)[0]
        self.num_baseline_actions = None

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        # bs = batch.batch_size
        # max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        obs = batch["obs"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]

        ## Train Critic

        # Calculate td-lambda targets
        critic_input = self.build_critic_inputs(batch["obs"], batch["actions"])
        target_qvals = self.target_critic(critic_input)
        targets = build_td_lambda_targets(rewards, terminated, mask, target_qvals, self.n_agents, self.args.gamma,
                                          self.args.td_lambda)

        critic_input = self.build_critic_inputs(obs, actions)
        qvals = self.critic(critic_input)
        td_error = (qvals - targets.detach())
        masked_td_error = td_error * mask # 0-out the targets that came from padded data

        ## Calculate Advantages

        actions_baseline_sampled = self.sample_actions(batch, n_actions=self.num_baseline_actions)
        n_actions_sampled = actions_baseline_sampled.shape[-1]
        actions_baseline = actions.unsqueeze(-1).unsqueeze(-1).repeat((1, 1, 1, 1, self.n_agents, n_actions_sampled))
        for i in range(self.n_agents):
            actions_baseline[:,:,i,0,i,:] = actions_baseline_sampled[:,:,i,:]
        actions_baseline = actions_baseline.permute(4, 5, 0, 1, 2, 3)
        obs_baseline = obs.unsqueeze(0).unsqueeze(0).repeat((self.n_agents, n_actions_sampled, 1, 1, 1, 1))
        critic_input_baseline = self.build_critic_inputs(
            obs_baseline.view(self.n_agents * n_actions_sampled * batch.batch_size, obs.shape[1], obs.shape[2], obs.shape[3]),
            actions_baseline.reshape(self.n_agents * n_actions_sampled * batch.batch_size, actions.shape[1], actions.shape[2], 1)
        )
        qvals_baseline_raw = self.critic(critic_input_baseline)
        qvals_baseline_raw = qvals_baseline_raw.view(self.n_agents, n_actions_sampled, batch.batch_size, *qvals_baseline_raw.shape[1:])
        qvals_baseline = self.baseline_agg(qvals_baseline_raw, dim=1)
        qvals_baseline = qvals_baseline.permute(1, 2, 0, 3) # B x T x N x 1
        advantages = qvals.unsqueeze(2).detach() - qvals_baseline.detach()

        ## Train Actor

        n_mask = mask.repeat(1, 1, self.n_agents).unsqueeze(-1)
        logits = self.eval_policy(batch)
        probs = torch.gather(logits, dim=-1, index=actions)
        probs[n_mask == 0] = 1.0
        log_probs = torch.log(probs)

        pi_loss = - ((advantages * log_probs) * n_mask).sum() / n_mask.sum()
        
        dist_entropy = Categorical(logits).entropy()
        dist_entropy[n_mask[:,:,:,0] == 0] = 0 # fill nan
        entropy_loss = (dist_entropy * n_mask).sum() / n_mask.sum()

        ## Optimise

        critic_loss = (masked_td_error ** 2).sum() / mask.sum()
        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        self.critic_optimiser.step()
        self.critic_training_steps += 1

        self.agent_optimiser.zero_grad()
        policy_loss = pi_loss - self.args.entropy * entropy_loss
        policy_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        ## Update Target

        if (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_step = self.critic_training_steps

        ## Stats

        if t_env - self.log_stats_t >= self.args.learner_log_interval:

            self.logger.log_stat("critic_loss", critic_loss.item(), t_env)
            self.logger.log_stat("critic_grad_norm", critic_loss.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", masked_td_error.abs().sum().item() / mask_elems, t_env)
            self.logger.log_stat("q_taken_mean", (qvals * mask).sum().item() / mask_elems, t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / mask_elems, t_env)

            self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("policy_loss", policy_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.logger.log_stat("pi_max", (logits.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env


    def eval_policy(self, batch):
        self.mac.init_hidden(batch.batch_size)
        return torch.stack([
                        self.mac.forward(ep_batch=batch, t=t)
                    for t in range(batch.max_seq_length - 1)], dim=1)

    def sample_actions(self, batch, n_actions):
        if n_actions is not None:
            actions = []
            for i in range(n_actions):
                self.mac.init_hidden(batch.batch_size)
                actions_sample = torch.stack([
                                    self.mac.select_actions(ep_batch=batch, t_ep=t, t_env=np.Inf)
                                for t in range(batch.max_seq_length - 1)], dim=1)
                actions.append(actions_sample)
            actions = torch.stack(actions, dim=-1)
            return actions # B x T x N x n_actions
        else:
            return torch.arange(self.n_actions)[None,None,None,:].expand(batch.batch_size,batch.max_seq_length-1,self.n_agents,-1)

    def build_critic_inputs(self, obs, actions):
        actions_onehot = self.onehot_transform.transform(actions)
        return torch.cat([obs, actions_onehot], dim=-1)

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        torch.save(self.critic.state_dict(), "{}/critic.th".format(path))
        torch.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        torch.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(torch.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(torch.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(torch.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
