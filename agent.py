import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

class VAE(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 2*latent_dim)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, obs_dim)
        )
        self.latent_dim = latent_dim
    def encode(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.enc(s)
        mu, logvar = h.chunk(2, dim=-1)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z, mu, logvar
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)
    def free_energy(self, s: torch.Tensor):
        z, mu, logvar = self.encode(s)
        recon = self.decode(z)
        recon_loss = F.mse_loss(recon, s, reduction='none').sum(-1)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        Fval = recon_loss + kl
        return Fval, {'recon': recon_loss.mean().item(), 'kl': kl.mean().item()}, z

def phi_obs(s: torch.Tensor) -> torch.Tensor:
    return s

def love_ego_cost(phi_s: torch.Tensor, v_vec: torch.Tensor, alpha: float = 1.0, ego_penalty: float = 0.0, gamma: float = 0.1) -> torch.Tensor:
    love = alpha * ((phi_s - v_vec)**2).sum(-1)
    ego = gamma * ego_penalty
    return love + ego

class Philosophy:
    def __init__(self, mode: str = 'none', params: Optional[Dict] = None, obs_tail_dim: int = 3):
        self.mode = mode
        self.params = params or {}
        self.obs_tail_dim = obs_tail_dim
        self.prev_probs: Optional[torch.Tensor] = None
    def buddhism_regularizer(self, F_now: torch.Tensor, F_next: torch.Tensor, s_next_hat: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        eta1 = float(self.params.get('eta1', 1.0))
        eta3 = float(self.params.get('eta3', 0.5))
        attach_thr = float(self.params.get('attach_thr', 0.8))
        self_store = s_next_hat[..., -self.obs_tail_dim]
        dF_pos = torch.clamp(F_next - F_now, min=0.0)
        attachment = torch.clamp(self_store - attach_thr, min=0.0)
        D_B = eta1 * dF_pos + eta3 * attachment
        return D_B.squeeze(), {'D_B': float(D_B.mean().item()), 'dF_pos': float(dF_pos.mean().item()), 'attachment': float(attachment.mean().item())}
    def stoic_regularizer(self, s_next_hat: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        zeta = float(self.params.get('zeta', 0.5))
        unctrl = s_next_hat[..., -self.obs_tail_dim+1:]
        cost = zeta * (unctrl**2).sum(-1)
        return cost.squeeze(), {'C_Stoic': float(cost.mean().item())}
    def existential_metrics(self, probs: torch.Tensor) -> Dict:
        metrics = {}
        if self.prev_probs is not None:
            p = torch.clamp(probs, 1e-8, 1.0)
            q = torch.clamp(self.prev_probs, 1e-8, 1.0)
            kl = (p * (p.log() - q.log())).sum().item()
            metrics['Resp_KL'] = float(kl)
        self.prev_probs = probs.detach().clone()
        return metrics

class Agent:
    def __init__(self, obs_dim: int, act_dim: int, device: str = None,
                 latent_dim: int = 32, lr: float = 3e-4,
                 alpha: float = 1.0, gamma: float = 0.1,
                 lambda_H: float = 0.3, lambda_X: float = 0.3, lambda_I: float = 1.0,
                 beta_temp: float = 5.0,
                 philosophy_mode: str = 'none', philosophy_params: Optional[Dict] = None,
                 obs_tail_dim: int = 3,
                 lambda_B: float = 0.5, lambda_S: float = 0.5, lambda_E: float = 0.3):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.gen = VAE(obs_dim, latent_dim).to(self.device)
        self.opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        self.v_vec = torch.zeros(obs_dim, device=self.device)
        self.alpha, self.gamma = alpha, gamma
        self.lambda_H, self.lambda_X, self.lambda_I = lambda_H, lambda_X, lambda_I
        self.act_dim = act_dim
        self.beta_temp = beta_temp
        self.prev_F = None
        self.valence = 0.0
        self.arousal = 0.0
        self.phil = Philosophy(philosophy_mode, philosophy_params, obs_tail_dim=obs_tail_dim)
        self.lambda_B = lambda_B
        self.lambda_S = lambda_S
        self.lambda_E = lambda_E
        self.last_metrics: Dict = {}
    @torch.no_grad()
    def plan(self, obs_i: np.ndarray, neigh_obs: List[np.ndarray], simulate=None, planner: str = None, planner_params: dict = None, agent_index: int = None) -> Tuple[int, np.ndarray]:
        s = torch.tensor(obs_i, dtype=torch.float32, device=self.device).unsqueeze(0)
        if planner == 'cem' and simulate is not None:
            cem = planner_params or {}
            return self._plan_cem(obs_i, neigh_obs, simulate, cem)
        phi_neighbors = []
        for o in neigh_obs:
            so = torch.tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0)
            phi_neighbors.append(phi_obs(so))
        logits = torch.zeros(self.act_dim, device=self.device)
        F_now,_,_ = self.gen.free_energy(s)
        for a in range(self.act_dim):
            z,_,_ = self.gen.encode(s)
            s_next_hat = self.gen.decode(z)
            phi_next = phi_obs(s_next_hat)
            prefer = love_ego_cost(phi_next, self.v_vec, alpha=self.alpha, ego_penalty=0.0, gamma=self.gamma)
            if len(phi_neighbors) > 0:
                sims = []
                for pn in phi_neighbors:
                    cos = F.cosine_similarity(phi_next, pn, dim=-1)
                    sims.append(1.0 - cos)
                H_cost = torch.stack(sims).mean()
            else:
                H_cost = torch.tensor(0.0, device=self.device)
            F_next,_,_ = self.gen.free_energy(s_next_hat)
            info_gain = torch.clamp(F_now - F_next, min=0.0)
            X_cost = torch.tensor(0.0, device=self.device)
            B_reg = torch.tensor(0.0, device=self.device)
            S_reg = torch.tensor(0.0, device=self.device)
            if self.phil.mode == 'buddhism':
                B_reg, m = self.phil.buddhism_regularizer(F_now, F_next, s_next_hat)
                self.last_metrics.update(m)
            elif self.phil.mode == 'stoic':
                S_reg, m = self.phil.stoic_regularizer(s_next_hat)
                self.last_metrics.update(m)
            G = prefer + self.lambda_H*H_cost + self.lambda_X*X_cost - self.lambda_I*info_gain + self.lambda_B*B_reg + self.lambda_S*S_reg
            logits[a] = -G.squeeze()
        # Existential regularization
        base_logits = self.beta_temp * logits
        probs = torch.softmax(base_logits, dim=-1)
        if self.phil.mode == 'existential' and self.lambda_E > 0.0:
            prev = self.phil.prev_probs if self.phil.prev_probs is not None else probs
            prev = torch.clamp(prev, 1e-8, 1.0)
            adj_logits = base_logits + self.lambda_E * torch.log(prev)
            probs = torch.softmax(adj_logits, dim=-1)
            self.last_metrics.update(self.phil.existential_metrics(probs))
        elif self.phil.mode == 'existential':
            self.last_metrics.update(self.phil.existential_metrics(probs))
        a0 = torch.multinomial(probs, 1).item()
        return a0, probs.detach().cpu().numpy()
    def update(self, obs_batch: np.ndarray):
        s = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
        Fval, logs, _ = self.gen.free_energy(s)
        loss = Fval.mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        F_mean = Fval.mean().item()
        if self.prev_F is not None:
            dF = F_mean - self.prev_F
            self.valence = float(np.tanh(-1.0 * dF))
            self.arousal = float(np.clip(abs(dF), 0, 1))
        self.prev_F = F_mean
        self.last_metrics.update({'F': F_mean, **logs})
        return {'F': F_mean, **logs}
    def _eval_sequence_G(self, s0: torch.Tensor, neigh_obs: List[np.ndarray], seq_obs: List[np.ndarray]):
        G_total = torch.tensor(0.0, device=self.device)
        F_prev,_,_ = self.gen.free_energy(s0)
        phi_neighbors = []
        for o in neigh_obs:
            so = torch.tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0)
            phi_neighbors.append(phi_obs(so))
        for obs_np in seq_obs:
            s = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            phi = phi_obs(s)
            prefer = love_ego_cost(phi, self.v_vec, alpha=self.alpha, ego_penalty=0.0, gamma=self.gamma)
            F_now,_,_ = self.gen.free_energy(s)
            info_gain = torch.clamp(F_prev - F_now, min=0.0)
            F_prev = F_now
            if len(phi_neighbors) > 0:
                sims = []
                for pn in phi_neighbors:
                    cos = F.cosine_similarity(phi, pn, dim=-1)
                    sims.append(1.0 - cos)
                H_cost = torch.stack(sims).mean()
            else:
                H_cost = torch.tensor(0.0, device=self.device)
            B_reg = torch.tensor(0.0, device=self.device)
            S_reg = torch.tensor(0.0, device=self.device)
            if self.phil.mode == 'buddhism':
                self_store = s[..., -3]
                attach_thr = float(self.phil.params.get('attach_thr', 0.8))
                eta3 = float(self.phil.params.get('eta3', 0.5))
                attachment = torch.clamp(self_store - attach_thr, min=0.0)
                B_reg = eta3 * attachment
            elif self.phil.mode == 'stoic':
                zeta = float(self.phil.params.get('zeta', 0.5))
                unctrl = s[..., -2:]
                S_reg = zeta * (unctrl**2).sum(-1)
            G = prefer + self.lambda_H*H_cost - self.lambda_I*info_gain + self.lambda_B*B_reg + self.lambda_S*S_reg
            G_total = G_total + G.squeeze()
        return G_total
    def _plan_cem(self, obs_i: np.ndarray, neigh_obs: List[np.ndarray], simulate, cem: dict):
        H = int(cem.get('H', 5)); N = int(cem.get('N', 64)); iters = int(cem.get('iters', 3)); elite_frac = float(cem.get('elite_frac', 0.2))
        act_dim = self.act_dim
        logits = torch.zeros(H, act_dim, device=self.device)
        s0 = torch.tensor(obs_i, dtype=torch.float32, device=self.device).unsqueeze(0)
        best_seq = None; best_score = torch.tensor(float('inf'), device=self.device)
        for _ in range(iters):
            seqs = []
            for n in range(N):
                acts = []
                for t in range(H):
                    probs_t = torch.softmax(logits[t], dim=-1)
                    a = torch.multinomial(probs_t, 1).item()
                    acts.append(a)
                seqs.append(acts)
            scores = []
            for acts in seqs:
                res = simulate(acts)
                if isinstance(res, tuple) and len(res)==2:
                    fut_obs, ext_list = res
                else:
                    fut_obs, ext_list = res, [0.0]*len(acts)
                G = self._eval_sequence_G(s0, neigh_obs, fut_obs) + self.lambda_X * torch.tensor(sum(ext_list), device=self.device)
                scores.append(G.item())
            idx = np.argsort(scores)
            elite_k = max(1, int(elite_frac * N))
            elite = [seqs[i] for i in idx[:elite_k]]
            elite_scores = [scores[i] for i in idx[:elite_k]]
            if elite_scores[0] < best_score.item():
                best_score = torch.tensor(elite_scores[0], device=self.device)
                best_seq = elite[0]
            new_logits = torch.zeros_like(logits)
            for t in range(H):
                counts = torch.zeros(act_dim, device=self.device)
                for seq in elite:
                    counts[seq[t]] += 1
                probs = counts / counts.sum().clamp_min(1.0)
                old_probs = torch.softmax(logits[t], dim=-1)
                probs = 0.5 * probs + 0.5 * old_probs
                new_logits[t] = torch.log(probs.clamp_min(1e-6))
            logits = new_logits
        first_probs = torch.softmax(logits[0], dim=-1)
        if self.phil.mode == 'existential' and self.lambda_E > 0.0:
            prev = self.phil.prev_probs if self.phil.prev_probs is not None else first_probs
            prev = torch.clamp(prev, 1e-8, 1.0)
            logits0_adj = logits[0] + self.lambda_E * torch.log(prev)
            first_probs = torch.softmax(logits0_adj, dim=-1)
        a0 = torch.multinomial(first_probs, 1).item() if best_seq is None else best_seq[0]
        if self.phil.mode == 'existential':
            self.last_metrics.update(self.phil.existential_metrics(first_probs))
        return a0, first_probs.detach().cpu().numpy()
