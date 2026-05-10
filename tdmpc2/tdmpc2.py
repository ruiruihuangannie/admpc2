import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from common.layers import api_model_conversion
from tensordict import TensorDict


class TDMPC2(torch.nn.Module):
	"""
	TD-MPC2 agent. Implements training + inference.
	Can be used for both single-task and multi-task experiments,
	and supports both state and pixel observations.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device('cuda:0')
		self.model = WorldModel(cfg).to(self.device)
		self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._termination.parameters() if self.cfg.episodic else []},
			{'params': self.model._Qs.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []
			 }
		], lr=self.cfg.lr, capturable=True)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda:0'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		print('Episode length:', cfg.episode_length)
		print('Discount factor:', self.discount)
		self._prev_mean = torch.nn.Buffer(torch.zeros(self._rollout_horizon, self.cfg.action_dim, device=self.device))
		self._last_plan_horizon = torch.nn.Buffer(torch.tensor(float(self.cfg.horizon), device=self.device))
		self._compile = cfg.compile and not cfg.adaptive_horizon
		if cfg.compile and cfg.adaptive_horizon:
			print('Disabling torch.compile because adaptive_horizon=true uses dynamic horizon selection.')
		if self._compile:
			print('Compiling update function with torch.compile...')
			self._update = torch.compile(self._update, mode="reduce-overhead")

	@property
	def plan(self):
		_plan_val = getattr(self, "_plan_val", None)
		if _plan_val is not None:
			return _plan_val
		if self._compile:
			plan = torch.compile(self._plan, mode="reduce-overhead")
		else:
			plan = self._plan
		self._plan_val = plan
		return self._plan_val

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	@property
	def _rollout_horizon(self):
		return self.cfg.h_max if self.cfg.adaptive_horizon else self.cfg.horizon

	def _select_horizon(self, depth_returns):
		"""Select rollout horizon from model-value inconsistency statistics."""
		if not self.cfg.adaptive_horizon:
			return self.cfg.horizon, {}
		depth_return_values = tuple(depth_returns.values())
		return_mean = torch.stack(depth_return_values).mean(dim=0)
		stats = {
			depth: (depth_return - return_mean).abs().mean()
			for depth, depth_return in depth_returns.items()
		}
		candidates = {
			depth: inconsistency
			for depth, inconsistency in stats.items()
			if self.cfg.h_min <= depth <= self.cfg.h_max
		}
		selected = min(candidates, key=candidates.get)
		selected = min(max(selected, self.cfg.h_min), self.cfg.h_max)
		return selected, stats

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.

		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		if isinstance(fp, dict):
			state_dict = fp
		else:
			state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
		state_dict = state_dict["model"] if "model" in state_dict else state_dict
		state_dict = api_model_conversion(self.model.state_dict(), state_dict)
		self.model.load_state_dict(state_dict)
		return

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Select an action by planning in the latent space of the world model.

		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (int): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)
		if self.cfg.mpc:
			return self.plan(obs, t0=t0, eval_mode=eval_mode, task=task).cpu()
		z = self.model.encode(obs, task)
		action, info = self.model.pi(z, task)
		if eval_mode:
			action = info["mean"]
		return action[0].cpu()

	@torch.no_grad()
	def _estimate_value(self, z, actions, task):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		horizon = self._rollout_horizon
		G, discount = 0, 1
		termination = torch.zeros(self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
		depth_returns = {}
		for t in range(horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
			z = self.model.next(z, actions[t], task)
			G = G + discount * (1-termination) * reward
			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update
			if self.cfg.episodic:
				termination = torch.clip(termination + (self.model.termination(z, task) > 0.5).float(), max=1.)
			depth = t + 1
			if self.cfg.adaptive_horizon and depth in (1, 3, 5, horizon):
				action, _ = self.model.pi(z, task)
				depth_returns[depth] = G + discount * (1-termination) * self.model.Q(z, action, task, return_type='avg')
		if self.cfg.adaptive_horizon:
			horizon, _ = self._select_horizon(depth_returns)
			self._last_plan_horizon.fill_(float(horizon))
			return depth_returns[horizon]
		action, _ = self.model.pi(z, task)
		return G + discount * (1-termination) * self.model.Q(z, action, task, return_type='avg')

	@torch.no_grad()
	def _plan(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Plan a sequence of actions using the learned world model.

		Args:
			z (torch.Tensor): Latent state from which to plan.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		# Sample policy trajectories
		horizon = self._rollout_horizon
		z = self.model.encode(obs, task)
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(horizon-1):
				pi_actions[t], _ = self.model.pi(_z, task)
				_z = self.model.next(_z, pi_actions[t], task)
			pi_actions[-1], _ = self.model.pi(_z, task)

		# Initialize state and parameters
		z = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(horizon, self.cfg.action_dim, device=self.device)
		std = torch.full((horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions

		# Iterate MPPI
		for _ in range(self.cfg.iterations):

			# Sample actions
			r = torch.randn(horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
			actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
			actions_sample = actions_sample.clamp(-1, 1)
			actions[:, self.cfg.num_pi_trajs:] = actions_sample
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# Compute elite actions
			value = self._estimate_value(z, actions, task).nan_to_num(0)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters
			max_value = elite_value.max(0).values
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score = score / score.sum(0)
			mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
			std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]

		# Select action
		rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
		actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
		a, std = actions[0], std[0]
		if not eval_mode:
			a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
		self._prev_mean.copy_(mean)
		return a.clamp(-1, 1)

	def update_pi(self, zs, task, replay_action=None, horizon=None):
		"""
		Update policy using a sequence of latent states.

		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		action, info = self.model.pi(zs, task)
		qs = self.model.Q(zs, action, task, return_type='avg', detach=True)
		self.scale.update(qs[0])
		qs = self.scale(qs)
		horizon = len(qs) if horizon is None else horizon
		action, qs = action[:horizon], qs[:horizon]

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"][:horizon] + qs).mean(dim=(1,2)) * rho).mean()
		behavior_reg_loss = torch.zeros((), device=self.device)
		if self.cfg.behavior_reg_coef > 0 and replay_action is not None:
			behavior_reg_loss = F.mse_loss(info["mean"][:horizon], replay_action[:horizon])
			pi_loss = pi_loss + self.cfg.behavior_reg_coef * behavior_reg_loss
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)

		info = TensorDict({
			"pi_loss": pi_loss,
			"pi_grad_norm": pi_grad_norm,
			"pi_entropy": info["entropy"][:horizon],
			"pi_scaled_entropy": info["scaled_entropy"][:horizon],
			"pi_scale": self.scale.value,
			"behavior_reg_loss": behavior_reg_loss,
		})
		return info

	@torch.no_grad()
	def _td_target(self, next_z, reward, terminated, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			terminated (torch.Tensor): Termination signal at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: TD-target.
		"""
		action, _ = self.model.pi(next_z, task)
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * (1-terminated) * self.model.Q(next_z, action, task, return_type='min', target=True)

	@torch.no_grad()
	def _estimate_depth_returns(self, zs, reward_preds, task):
		"""Estimate returns from the same latent state at selected rollout depths."""
		horizon = self._rollout_horizon
		depths = tuple(dict.fromkeys(
			depth for depth in (1, 3, 5, horizon)
			if depth <= horizon
		))
		G = torch.zeros_like(reward_preds[0, :, :1])
		discount = torch.ones_like(G)
		discount_update = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		returns = {}
		for t in range(horizon):
			G = G + discount * math.two_hot_inv(reward_preds[t], self.cfg)
			discount = discount * discount_update
			depth = t + 1
			if depth in depths:
				bootstrap_action, _ = self.model.pi(zs[depth], task)
				bootstrap_value = self.model.Q(zs[depth], bootstrap_action, task, return_type='avg', detach=True)
				returns[depth] = G + discount * bootstrap_value
		return returns

	def _update(self, obs, action, reward, terminated, task=None):
		# Compute targets
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target(next_z, reward, terminated, task)

		# Prepare for update
		self.model.train()

		# Latent rollout
		horizon = self._rollout_horizon
		zs = torch.empty(horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self.model.encode(obs[0], task)
		zs[0] = z
		consistency_losses = []
		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			z = self.model.next(z, _action, task)
			consistency_losses.append(F.mse_loss(z, _next_z) * self.cfg.rho**t)
			zs[t+1] = z

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, task, return_type='all')
		reward_preds = self.model.reward(_zs, action, task)
		if self.cfg.episodic:
			termination_pred = self.model.termination(zs[1:], task, unnormalized=True)

		# Compute losses
		reward_losses, value_losses = [], []
		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_losses.append(math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t)
			value_loss = 0
			for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
				value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t
			value_losses.append(value_loss)

		depth_returns = self._estimate_depth_returns(zs.detach(), reward_preds.detach(), task)
		selected_horizon, inconsistency_stats = self._select_horizon(depth_returns)
		consistency_loss = sum(consistency_losses[:selected_horizon]) / selected_horizon
		reward_loss = sum(reward_losses[:selected_horizon]) / selected_horizon
		if self.cfg.episodic:
			termination_loss = F.binary_cross_entropy_with_logits(
				termination_pred[:selected_horizon],
				terminated[:selected_horizon],
			)
		else:
			termination_loss = 0.
		value_loss = sum(value_losses[:selected_horizon]) / (selected_horizon * self.cfg.num_q)
		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.termination_coef * termination_loss +
			self.cfg.value_coef * value_loss
		)

		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		# Update policy
		pi_horizon = selected_horizon if self.cfg.adaptive_horizon else None
		pi_info = self.update_pi(zs.detach(), task, replay_action=action, horizon=pi_horizon)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		q_values = math.two_hot_inv(qs, self.cfg)
		depth_return_values = tuple(depth_returns.values())
		return_mean = torch.stack(depth_return_values).mean(dim=0)
		info = TensorDict({
			"rollout_horizon": torch.tensor(float(selected_horizon), device=self.device),
			"plan_rollout_horizon": self._last_plan_horizon,
			"horizon_at_h_min": torch.tensor(float(selected_horizon == self.cfg.h_min), device=self.device),
			"horizon_at_h_max": torch.tensor(float(selected_horizon == self.cfg.h_max), device=self.device),
			"consistency_loss": consistency_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"value_pred_mean": q_values.mean(),
			"value_pred_std": q_values.std(unbiased=False),
			"value_pred_min": q_values.min(),
			"value_pred_max": q_values.max(),
			"value_overestimation_proxy": q_values.mean() - depth_returns[horizon].mean(),
			"termination_loss": termination_loss,
			"total_loss": total_loss,
			"grad_norm": grad_norm,
		})
		for depth, depth_return in depth_returns.items():
			info[f"return_estimate_depth_{depth}"] = depth_return.mean()
			info[f"model_value_inconsistency_depth_{depth}"] = inconsistency_stats.get(depth, (depth_return - return_mean).abs().mean())
		info["return_estimate_depth_h"] = depth_returns[horizon].mean()
		info["model_value_inconsistency_depth_h"] = inconsistency_stats.get(horizon, (depth_returns[horizon] - return_mean).abs().mean())
		info["model_value_inconsistency"] = torch.stack(depth_return_values).std(dim=0, unbiased=False).mean()
		if self.cfg.episodic:
			info.update(math.termination_statistics(torch.sigmoid(termination_pred[-1]), terminated[-1]))
		info.update(pi_info)
		return info.detach().mean()

	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		obs, action, reward, terminated, task = buffer.sample()
		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		torch.compiler.cudagraph_mark_step_begin()
		return self._update(obs, action, reward, terminated, **kwargs)
