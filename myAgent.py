import copy
from dataclasses import dataclass, asdict, is_dataclass, field

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim

from myActivator import tanhAndScale


# DDPGエージェント
class DDPGAgent:
    def __init__(self,Config,device=None):
        if Config:
            self.Config = Config
        else:
            raise ValueError("No Config!!")
        
        # ---- device 決定（指定がなければ CUDA があれば CUDA）----
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ---- action bounds（device を揃えるため Tensor 化）----
        self.u_low = torch.as_tensor(Config.u_llim, dtype=torch.float32, device=self.device)
        self.u_high = torch.as_tensor(Config.u_ulim, dtype=torch.float32, device=self.device)
        
        self.Q_net = self.build_net(
            Config.Q_net_in,
            Config.Q_net_sizes,
            Config.Q_net_out,
            ).to(self.device)
        self.Q_net.train()

        self.P_net = self.build_net(
            Config.P_net_in,
            Config.P_net_sizes,
            Config.P_net_out,
            tanhAndScale(a_high=self.u_high,a_low=self.u_low),
            ).to(self.device)
        self.P_net.train()

        # ---- Target nets（重要：deepcopy で別物を作る）----
        self.Q_target_net = copy.deepcopy(self.Q_net).to(self.device)
        self.P_target_net = copy.deepcopy(self.P_net).to(self.device)
        self.Q_target_net.eval()
        self.P_target_net.eval()

        self.Q_optim = optim.Adam(self.Q_net.parameters(),lr=Config.Q_lr)
        self.P_optim = optim.Adam(self.P_net.parameters(),lr=Config.P_lr)

    def to(self, device):
        """エージェント内部のネットと必要Tensorを指定 device に移す。"""
        self.device = torch.device(device)
        self.Q_net.to(self.device)
        self.P_net.to(self.device)
        self.Q_target_net.to(self.device)
        self.P_target_net.to(self.device)
        self.u_low = self.u_low.to(self.device)
        self.u_high = self.u_high.to(self.device)
        return self


    def build_net(self, input_size, hidden_sizes, output_size=1, output_activator=None):
        layers = []
        for input_size, output_size in zip([input_size]+hidden_sizes, hidden_sizes+[output_size]):
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ReLU())
        layers = layers[:-1]  # 最後のReLUだけ取り除く
        if output_activator:
            layers.append(output_activator)
        net = nn.Sequential(*layers)
        return net
    

    @torch.no_grad()
    def step(self, observation) -> np.ndarray:
        """ノイズなし（評価用）。環境に渡す行動を返す。"""
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)

        action = self.P_net(obs_t)
        action = torch.clamp(action, self.u_low, self.u_high)
        return action.squeeze(0).cpu().numpy()
    

    @torch.no_grad()
    def step_with_noise(self, observation):
        # 1) observation を Tensor にし、ネットと同じ device に載せる
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # (obs_dim,) -> (1, obs_dim)

        # 2) 決定論的行動 a = μθ(s)
        action = self.P_net(obs)  # shape: (1, act_dim) を想定

        # 3) ε ~ N(0, σ^2 I) を生成して加算（探索）
        eps = float(self.Config.sig) * torch.randn_like(action)
        action = action + eps

        # 4) 出力制約 [u_low, u_high] に収める（安全弁）
        action = torch.clamp(action, self.u_low, self.u_high)

        # 5) 環境に渡すならバッチ次元を落として返す（numpy が必要なら .cpu().numpy()）
        return action.squeeze(0).cpu().numpy()
    

    def save_all(self, path: str, extra: dict | None = None):
        """
        Actor/Critic + target nets をまとめて保存（最終モデル用）。
        """
        cfg = asdict(self.Config) if is_dataclass(self.Config) else self.Config

        ckpt = {
            "config": cfg,
            "P_net": self.P_net.state_dict(),
            "Q_net": self.Q_net.state_dict(),
            "P_target_net": self.P_target_net.state_dict(),
            "Q_target_net": self.Q_target_net.state_dict(),
        }
        if extra is not None:
            ckpt["extra"] = extra

        torch.save(ckpt, path)


    def load_all(self, path: str, map_location=None):
        """
        save_all() で保存したチェックポイントをロード。

        PyTorch 2.6 以降:
        torch.load() のデフォルトが weights_only=True になったため、
        config/extra を含むチェックポイントはそのままだと UnpicklingError になり得る。
        その回避として「信頼できるチェックポイントに限り」 weights_only=False を明示する。

        ※ map_location は "cpu" や device を指定可。
        """
        # PyTorch 2.6+ では weights_only 引数がある
        try:
            ckpt = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            # 古い PyTorch（weights_only 引数が無い）向け
            ckpt = torch.load(path, map_location=map_location)

        self.P_net.load_state_dict(ckpt["P_net"])
        self.Q_net.load_state_dict(ckpt["Q_net"])
        self.P_target_net.load_state_dict(ckpt["P_target_net"])
        self.Q_target_net.load_state_dict(ckpt["Q_target_net"])

        return ckpt.get("extra", None)
    

    def mode2eval(self):
        self.P_net.eval()
        self.Q_net.eval()


    def mode2train(self):
        self.P_net.train()
        self.Q_net.train()
    

    @torch.no_grad()
    def soft_update(self, target_net, online_net, tau):
        """
        Polyak averaging:
          θ' ← (1-τ) θ' + τ θ
        """
        for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
            target_param.mul_(1.0 - tau).add_(tau * online_param)

    
    def update_net(self,states,actions,rewards,states_next,dones=None):
        """
        1回の更新（Critic→Actor→Target soft update）
        戻り値： (q_loss, p_loss) のスカラー
        """
        # ---- minibatch を device 上 Tensor に統一 ----
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        states_next = torch.as_tensor(states_next, dtype=torch.float32, device=self.device)

        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)

        if dones is None:
            dones = torch.zeros((states.shape[0], 1), dtype=torch.float32, device=self.device)
        else:
            dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
            if dones.dim() == 1:
                dones = dones.unsqueeze(1)

        with torch.no_grad():
            actions_next_for_target = self.P_target_net(states_next)
            y_targets = rewards + self.Config.gamma*(1-dones)*self.Q_target_net(torch.cat([states_next, actions_next_for_target], dim=1))
        
        Q_values = self.Q_net(torch.cat([states,actions],dim=1))
        Q_loss = F.mse_loss(y_targets,Q_values)
        self.Q_optim.zero_grad()
        Q_loss.backward()
        self.Q_optim.step()

        # ---- Actor update ----
        # Actor 更新では Q_net を通すが、Q_net 自体は更新しないので凍結（計算の節約＋安全）
        for p in self.Q_net.parameters():
            p.requires_grad_(False)
        
        actions_for_Ploss = self.P_net(states)
        P_loss = -self.Q_net(torch.cat([states,actions_for_Ploss],dim=1)).mean()
        self.P_optim.zero_grad()
        P_loss.backward()
        self.P_optim.step()

        for p in self.Q_net.parameters():
            p.requires_grad_(True)

        self.soft_update(
            target_net=self.Q_target_net,
            online_net=self.Q_net,
            tau=self.Config.tau
            )
        self.soft_update(
            target_net=self.P_target_net,
            online_net=self.P_net,
            tau=self.Config.tau
            )
        
        return float(Q_loss.item()), float(P_loss.item())
    


# TD3エージェント
class TD3Agent:
    def __init__(self,Config,device=None):
        if Config:
            self.Config = Config
        else:
            raise ValueError("No Config!!")
        
        # ---- device 決定（CUDAがあればCUDA、指示が無ければ）----
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ---- action bounds（deviceを揃える為 Tensor 化）----
        self.u_low = torch.as_tensor(Config.u_llim, dtype=torch.float32, device=self.device)
        self.u_high = torch.as_tensor(Config.u_ulim, dtype=torch.float32, device=self.device)

        self.Q_net1 = self.build_net(
            Config.Q_net_in,
            Config.Q_net_sizes,
            Config.Q_net_out,
        ).to(self.device)
        self.Q_net1.train()

        self.Q_net2 = self.build_net(
            Config.Q_net_in,
            Config.Q_net_sizes,
            Config.Q_net_out,
        ).to(self.device)
        self.Q_net2.train()

        self.P_net = self.build_net(
            Config.P_net_in,
            Config.P_net_sizes,
            Config.P_net_out,
            tanhAndScale(a_high=self.u_high,a_low=self.u_low),
        ).to(self.device)
        self.P_net.train()

        # ---- Target nets ----
        self.Q_target_net1 = copy.deepcopy(self.Q_net1).to(self.device)
        self.Q_target_net2 = copy.deepcopy(self.Q_net2).to(self.device)
        self.P_target_net = copy.deepcopy(self.P_net).to(self.device)
        self.Q_target_net1.eval()
        self.Q_target_net2.eval()
        self.P_target_net.eval()

        self.Q_optim1 = optim.Adam(self.Q_net1.parameters(), lr=Config.Q_lr1)
        self.Q_optim2 = optim.Adam(self.Q_net2.parameters(), lr=Config.Q_lr2)
        self.P_optim = optim.Adam(self.P_net.parameters(), lr=Config.P_lr)

    def to(self, device):
        """エージェントの内部のネットと必要Tensorを全部指定 device に移す"""
        self.device = torch.device(device)
        self.Q_net1.to(self.device)
        self.Q_net2.to(self.device)
        self.P_net.to(self.device)
        self.Q_target_net1.to(self.device)
        self.Q_target_net2.to(self.device)
        self.P_target_net.to(self.device)
        self.u_low = self.u_low.to(self.device)
        self.u_high = self.u_high.to(self.device)
        return self

    def build_net(self, input_size, hidden_sizes, output_size=1, output_activator=None):
        layers = []
        for input_size, output_size in zip([input_size]+hidden_sizes, hidden_sizes+[output_size]):
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ReLU())
        layers = layers[:-1]  # 最後のReLUだけ取り除く
        if output_activator:
            layers.append(output_activator)
        net = nn.Sequential(*layers)
        return net
    
    @torch.no_grad()
    def step(self, observation):
        """ノイズ無し（評価用）。環境に渡す行動を返す。"""
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        
        action = self.P_net(obs_t)
        action = torch.clamp(action, self.u_low, self.u_high)
        return action.squeeze(0).cpu().numpy()
    
    @torch.no_grad()
    def step_with_noise(self, observation):
        # observationをTensorにし、deviceに送る
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        # 決定論的行動決定
        action = self.P_net(obs)  # shape: (1, act_dim)

        # ε ~ N(0, σ^2 I) を生成して加算（探索ノイズを加える）
        eps = float(self.Config.sig) * torch.rand_like(action)
        action = action + eps

        # 出力製薬 [u_low, u_high] 以内に収める
        action = torch.clamp(action, self.u_low, self.u_high)

        # 環境に渡すならバッチ次元を落として渡す
        return action.squeeze(0).cpu().numpy()
    
    def save_all(self, path: str, extra: dict | None = None):
        """
        Actor/Critic + target nets をまとめて保存（最終モデル用）。
        """
        cfg = asdict(self.Config) if is_dataclass(self.Config) else self.Config

        ckpt = {
            "config": cfg,
            "P_net": self.P_net.state_dict(),
            "Q_net1": self.Q_net1.state_dict(),
            "Q_net2": self.Q_net2.state_dict(),
            "P_target_net": self.P_target_net.state_dict(),
            "Q_target_net1": self.Q_target_net1.state_dict(),
            "Q_target_net2": self.Q_target_net2.state_dict(),
        }
        if extra is not None:
            ckpt["extra"] = extra

        torch.save(ckpt, path)


    def load_all(self, path: str, map_location=None):
        """
        save_all() で保存したチェックポイントをロード。

        PyTorch 2.6 以降:
        torch.load() のデフォルトが weights_only=True になったため、
        config/extra を含むチェックポイントはそのままだと UnpicklingError になり得る。
        その回避として「信頼できるチェックポイントに限り」 weights_only=False を明示する。

        ※ map_location は "cpu" や device を指定可。
        """
        # PyTorch 2.6+ では weights_only 引数がある
        try:
            ckpt = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            # 古い PyTorch（weights_only 引数が無い）向け
            ckpt = torch.load(path, map_location=map_location)

        self.P_net.load_state_dict(ckpt["P_net"])
        self.Q_net1.load_state_dict(ckpt["Q_net1"])
        self.Q_net2.load_state_dict(ckpt["Q_net2"])
        self.P_target_net.load_state_dict(ckpt["P_target_net"])
        self.Q_target_net1.load_state_dict(ckpt["Q_target_net1"])
        self.Q_target_net2.load_state_dict(ckpt["Q_target_net2"])

        return ckpt.get("extra", None)
    
    def mode2eval(self):
        self.P_net.eval()
        self.Q_net1.eval()
        self.Q_net2.eval()

    def mode2train(self):
        self.P_net.train()
        self.Q_net1.train()
        self.Q_net2.train()

    @torch.no_grad()
    def soft_update(self, target_net, online_net, tau):
        """
        Polyak averaging:
          θ' ← (1-τ) θ' + τ θ
        """
        for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
            target_param.mul_(1.0-tau).add_(tau*online_param)

    def update_net(self, states, actions, rewards, states_next, dones=None):
        """
        1回の更新（Critic→Actor→Target soft update）
        戻り値： (q_loss, p_loss) のスカラー
        """
        # ---- minibatch を device 上 Tensor に統一 ----
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        states_next = torch.as_tensor(states_next, dtype=torch.float32, device=self.device)

        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        
        if dones is None:
            dones = torch.zeros((states.shape[0],1), dtype=torch.float32, device=self.device)
        else:
            dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
            if dones.dim() == 1:
                dones = dones.unsqueeze(1)
        
        with torch.no_grad():
            actions_next_for_target = self.P_target_net(states_next)
            Q_target1 = self.Q_target_net1(torch.cat([states_next,actions_next_for_target],dim=1))
            Q_target2 = self.Q_target_net2(torch.cat([states_next,actions_next_for_target],dim=1))
            y_targets = rewards + self.Config.gamma*(1-dones)*torch.min(Q_target1,Q_target2)

        Q_values1 = self.Q_net1(torch.cat([states,actions],dim=1))
        Q_values2 = self.Q_net2(torch.cat([states,actions],dim=1))
        Q_loss1 = F.mse_loss(y_targets,Q_values1)
        Q_loss2 = F.mse_loss(y_targets,Q_values2)
        self.Q_optim1.zero_grad()
        self.Q_optim2.zero_grad()
        Q_loss1.backward()
        Q_loss2.backward()
        self.Q_optim1.step()
        self.Q_optim2.step()

        # ---- Actor update ----
        # Actor 更新では Q_net を通すが、Q_net自体は更新しないので凍結1
        for p in self.Q_net1.parameters():
            p.requires_grad_(False)

        actions_for_Ploss = self.P_net(states)
        P_loss = -self.Q_net1(torch.cat([states,actions_for_Ploss],dim=1)).mean()
        self.P_optim.zero_grad()
        P_loss.backward()
        self.P_optim.step()

        for p in self.Q_net1.parameters():
            p.requires_grad_(True)

        # ---- Target net update ----
        self.soft_update(
            target_net=self.Q_target_net1,
            online_net=self.Q_net1,
            tau=self.Config.tau,
        )
        self.soft_update(
            target_net=self.Q_target_net2,
            online_net=self.Q_net2,
            tau=self.Config.tau,
        )
        self.soft_update(
            target_net=self.P_target_net,
            online_net=self.P_net,
            tau=self.Config.tau
        )

        return float(Q_loss1.item()), float(Q_loss2.item()), float(P_loss.item())
    


# TRPOエージェント
class TRPOAgent:
    """
    Baseline（動いてる版）に寄せた TRPO 実装（修正済み完全版）。

    修正点:
    1. step() メソッドをデフォルトで「決定論的（平均値）」に変更し、評価時のスコアを安定化。
    2. update_net() で actions の形状を (Batch, act_dim) に強制し、Broadcasting事故を防止。
    3. update_net() で log_probs がリスト(float)かTensorか判別し、確実に計算グラフから切断。
    """

    def __init__(self, Config, device=None):
        if Config is None:
            raise ValueError("No Config!!")
        self.Config = Config

        # device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # action bounds (for env clip only)
        self.u_low = torch.as_tensor(Config.u_llim, dtype=torch.float32, device=self.device)
        self.u_high = torch.as_tensor(Config.u_ulim, dtype=torch.float32, device=self.device)

        # networks
        self.V_net = self.build_net(Config.V_net_in, Config.V_net_sizes, Config.V_net_out).to(self.device)
        self.P_net = self.build_net(Config.P_net_in, Config.P_net_sizes, Config.P_net_out).to(self.device)

        self.V_net.train()
        self.P_net.train()

        # log_std は状態に依存しないパラメータ
        action_dim = Config.P_net_out
        self.log_std = nn.Parameter(torch.zeros(action_dim, device=self.device))

        # critic optimizer（baselineはAdam）
        self.V_optim = optim.Adam(self.V_net.parameters(), lr=Config.V_lr)

        # hyperparams
        self.gamma = float(Config.gamma)
        self.tau = float(Config.lam)  # baselineの TAU (= GAE lambda)
        self.max_kl = float(Config.max_kl)
        self.cg_iters = int(Config.cg_iters)
        self.cg_damping = float(Config.cg_damping)

        self.value_train_iters = int(getattr(Config, "value_train_iters", 5))
        self.value_l2_reg = float(getattr(Config, "value_l2_reg", 1e-3))

        self.backtrack_coeff = float(getattr(Config, "ls_backtrack", 0.8))
        self.backtrack_iters = int(getattr(Config, "ls_max_steps", 10))

    def build_net(self, input_size, hidden_sizes, output_size):
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, output_size))
        return nn.Sequential(*layers)

    # ------------------------------------------------------------
    # Policy helpers (baseline-style)
    # ------------------------------------------------------------
    def _policy_mean(self, states: torch.Tensor) -> torch.Tensor:
        # (T, obs_dim) -> (T, act_dim)
        return self.P_net(states)

    def _policy_dist(self, states: torch.Tensor) -> Normal:
        mean = self._policy_mean(states)
        std = torch.exp(self.log_std)  # (act_dim,)
        # broadcasting: (T, act_dim) with (act_dim,)
        return Normal(mean, std)

    def _log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # returns shape (T,)
        dist = self._policy_dist(states)
        return dist.log_prob(actions).sum(dim=-1)

    @torch.no_grad()
    def get_action_and_log_prob(self, state, deterministic=False):
        """
        deterministic: Trueなら平均値(mean)を返す。Falseならサンプリング。
        """
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if s.dim() == 1:
            s = s.unsqueeze(0)  # (1, obs_dim)

        dist = self._policy_dist(s)  # Normal(mean, std)

        if deterministic:
            a = dist.mean  # (1, act_dim)
            logp = None
        else:
            a = dist.sample()
            logp = dist.log_prob(a).sum(dim=-1)  # (1,)

        # 返す action は “clip前” を返す（baselineと同じ）
        # envに入れる直前で clip してください
        a = a.squeeze(0)
        if logp is not None:
            logp = logp.squeeze(0)
        return a, logp

    @torch.no_grad()
    def step(self, state):
        """
        【重要修正】推論時はデフォルトで「決定論的（平均値）」を使用する。
        これにより、評価時のスコアが安定して高くなる。
        """
        a, _ = self.get_action_and_log_prob(state, deterministic=True)
        return a.cpu().numpy()

    # ------------------------------------------------------------
    # Flat params / grads (baseline-style)
    # ------------------------------------------------------------
    def _flat_params(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in list(self.P_net.parameters()) + [self.log_std]])

    def _set_flat_params(self, flat: torch.Tensor):
        idx = 0
        # P_net params
        for p in self.P_net.parameters():
            n = p.numel()
            p.data.copy_(flat[idx:idx+n].view_as(p))
            idx += n
        # log_std
        n = self.log_std.numel()
        self.log_std.data.copy_(flat[idx:idx+n].view_as(self.log_std))
        idx += n

    def _flat_grad(self, scalar: torch.Tensor, retain_graph=False, create_graph=False) -> torch.Tensor:
        params = list(self.P_net.parameters()) + [self.log_std]
        grads = torch.autograd.grad(
            scalar, params,
            retain_graph=retain_graph,
            create_graph=create_graph,
            allow_unused=False,
        )
        return torch.cat([g.contiguous().view(-1) for g in grads])

    # ------------------------------------------------------------
    # GAE (baseline-style)
    # ------------------------------------------------------------
    @torch.no_grad()
    def _compute_gae(self, rewards, values, next_values, dones):
        """
        rewards, dones: (T,)
        values, next_values: (T,)
        """
        T = rewards.shape[0]
        adv = torch.zeros_like(rewards)
        gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                nv = next_values[t]
            else:
                nv = values[t + 1]
            delta = rewards[t] + self.gamma * nv * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.tau * (1.0 - dones[t]) * gae
            adv[t] = gae

        ret = adv + values
        return adv, ret

    # ------------------------------------------------------------
    # TRPO core (baseline-style)
    # ------------------------------------------------------------
    def _conjugate_gradient(self, Avp_fn, b, n_iters=10, residual_tol=1e-10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)

        for _ in range(n_iters):
            Ap = Avp_fn(p)
            alpha = rdotr / (torch.dot(p, Ap) + 1e-8)
            x = x + alpha * p
            r = r - alpha * Ap
            new_rdotr = torch.dot(r, r)
            if new_rdotr < residual_tol:
                break
            beta = new_rdotr / (rdotr + 1e-12)
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def _fisher_vector_product(self, states: torch.Tensor, v: torch.Tensor):
        dist_new = self._policy_dist(states)

        # detach old
        mean_old = dist_new.mean.detach()
        std_old = dist_new.stddev.detach()
        dist_old = Normal(mean_old, std_old)

        # KL(old||new)
        kl = torch.distributions.kl_divergence(dist_old, dist_new).sum(dim=-1).mean()

        # grad KL
        kl_grad = self._flat_grad(kl, retain_graph=True, create_graph=True)

        # (grad KL)^T v
        kl_grad_v = torch.dot(kl_grad, v)

        # Hessian-vector product
        hvp = self._flat_grad(kl_grad_v, retain_graph=True, create_graph=False)

        return hvp + self.cg_damping * v

    def _surrogate_loss(self, states, actions, advantages, old_log_probs):
        new_logp = self._log_prob(states, actions)
        ratio = torch.exp(new_logp - old_log_probs)
        return -(ratio * advantages).mean()

    def _trpo_step(self, states, actions, advantages, old_log_probs):
        # 1) policy gradient of surrogate loss
        loss = self._surrogate_loss(states, actions, advantages, old_log_probs)
        g = self._flat_grad(loss, retain_graph=True, create_graph=False)

        # 2) CG: solve F x = -g
        def Fvp(v):
            return self._fisher_vector_product(states, v)

        step_dir = self._conjugate_gradient(Fvp, -g, n_iters=self.cg_iters)

        # 3) scale to satisfy KL constraint
        shs = 0.5 * torch.dot(step_dir, Fvp(step_dir))
        if shs.item() <= 0.0:
            return False

        lm = torch.sqrt(shs / self.max_kl)
        full_step = step_dir / (lm + 1e-8)

        # 4) line search
        old_params = self._flat_params()
        old_loss = loss.item()

        step_frac = 1.0
        for _ in range(self.backtrack_iters):
            new_params = old_params + step_frac * full_step
            self._set_flat_params(new_params)

            with torch.no_grad():
                new_loss = self._surrogate_loss(states, actions, advantages, old_log_probs).item()

            if new_loss < old_loss:
                return True

            step_frac *= self.backtrack_coeff

        # fail: revert
        self._set_flat_params(old_params)
        return False

    # ------------------------------------------------------------
    # Value update (baseline-style)
    # ------------------------------------------------------------
    def _update_value_function(self, states, returns):
        last_loss = None
        for _ in range(self.value_train_iters):
            v_pred = self.V_net(states).squeeze(-1)
            v_loss = (v_pred - returns).pow(2).mean()

            # L2 reg
            l2 = 0.0
            for p in self.V_net.parameters():
                l2 = l2 + p.pow(2).sum()
            v_loss = v_loss + self.value_l2_reg * l2

            self.V_optim.zero_grad()
            v_loss.backward()
            self.V_optim.step()

            last_loss = v_loss.item()
        return last_loss

    # ------------------------------------------------------------
    # Public update API (Fixed Version)
    # ------------------------------------------------------------
    def update_net(self, states, actions, log_probs, rewards, states_next, dones):
        """
        学習ループから呼ばれるメイン更新関数。

        Fixes:
        1. actions.view(-1, act_dim) で形状不一致バグを修正。
        2. old_log_probs の detach 漏れを修正。
        """
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        states_next = torch.as_tensor(states_next, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).view(-1)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).view(-1)

        # 【修正】Actionの次元を明示的に整形 (Broadcasting事故防止)
        # 入力が (Batch,) だと broadcasting で (Batch, Batch) になり計算が壊れるため、(Batch, 1) 等に強制する
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device).view(-1, self.Config.act_dim)

        # 【修正】old_log_probs を確実に計算グラフから切断 (.detach())
        if isinstance(log_probs, (list, tuple)):
            # floatのリストが来た場合 (推奨) -> Tensor化
            old_log_probs = torch.tensor(log_probs, dtype=torch.float32, device=self.device).view(-1)
        else:
            # Tensorが来た場合 -> detachしてコピー
            old_log_probs = torch.as_tensor(log_probs, dtype=torch.float32, device=self.device).view(-1).detach()

        # 1) values
        with torch.no_grad():
            values = self.V_net(states).squeeze(-1)
            next_values = self.V_net(states_next).squeeze(-1)

        # 2) GAE
        with torch.no_grad():
            advantages, returns = self._compute_gae(rewards, values, next_values, dones)
            # normalize advantage
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3) TRPO policy update
        trpo_ok = self._trpo_step(states, actions, advantages, old_log_probs)

        # 4) Value update
        v_loss = self._update_value_function(states, returns)

        return {"V_loss": v_loss, "trpo_ok": trpo_ok}

    # ------------------------------------------------------------
    # misc
    # ------------------------------------------------------------
    def to(self, device):
        self.device = torch.device(device)
        self.V_net.to(self.device)
        self.P_net.to(self.device)
        self.log_std.data = self.log_std.data.to(self.device)
        self.u_low = self.u_low.to(self.device)
        self.u_high = self.u_high.to(self.device)
        return self

    def mode2eval(self):
        self.V_net.eval()
        self.P_net.eval()

    def mode2train(self):
        self.V_net.train()
        self.P_net.train()

    def save_all(self, path: str, extra: dict | None = None):
        cfg = asdict(self.Config) if is_dataclass(self.Config) else self.Config
        save_dict = {
            "Config": cfg,
            "V_net_state_dict": self.V_net.state_dict(),
            "P_net_state_dict": self.P_net.state_dict(),
            "log_std": self.log_std.data,
        }
        if extra is not None:
            save_dict.update(extra)
        torch.save(save_dict, path)
        
    def load_all(self, path: str, map_location=None):
        load_dict = torch.load(path, map_location=map_location)
        self.V_net.load_state_dict(load_dict["V_net_state_dict"])
        self.P_net.load_state_dict(load_dict["P_net_state_dict"])
        self.log_std.data = load_dict["log_std"].to(self.device)