import numpy as np
import torch
from public_goods_env import PublicGoodsGrid, PGConfig
from agent import Agent

def neighbor_indices(env: PublicGoodsGrid, i: int):
    x,y = env.pos[i]
    idxs = []
    for j in range(env.n_agents):
        if j==i: continue
        xj,yj = env.pos[j]
        if abs(x-xj)+abs(y-yj) <= 2:
            idxs.append(j)
    return idxs

def make_simulator(env: PublicGoodsGrid, agent_index: int, other_policy: str = 'stay', mean_field_probs: np.ndarray = None):
    def simulate(acts_seq: list):
        sim_env = env.clone(seed=12345 + env.t + agent_index)
        _ = sim_env._obs_all()
        obs_list = []; ext_list = []
        for a in acts_seq:
            joint = []
            for j in range(sim_env.n_agents):
                if j == agent_index:
                    joint.append(a)
                else:
                    if other_policy == 'stay':
                        joint.append(0)
                    elif other_policy == 'random':
                        joint.append(int(sim_env.rng.integers(0, sim_env.act_dim)))
                    elif other_policy == 'mean_field' and mean_field_probs is not None:
                        p = mean_field_probs / (mean_field_probs.sum() + 1e-6)
                        aj = sim_env.rng.choice(len(p), p=p)
                        joint.append(aj)
                    else:
                        joint.append(0)
            next_obs, _, _, info = sim_env.step(joint)
            obs_list.append(next_obs[agent_index])
            ext_net = float(info['ext_net'][agent_index]) if 'ext_net' in info else 0.0
            ext_list.append(ext_net)
        return obs_list, ext_list
    return simulate

def _make_agents(env: PublicGoodsGrid, philosophy_mode: str = 'none', philosophy_params=None, agent_kwargs=None):
    obs_dim = env.obs_dim(); agent_kwargs = agent_kwargs or {}
    return [Agent(obs_dim=obs_dim, act_dim=env.act_dim,
                  philosophy_mode=philosophy_mode, philosophy_params=philosophy_params,
                  **agent_kwargs) for _ in range(env.n_agents)]

def run_episode(steps=300, cfg: PGConfig = PGConfig(), learn=True,
                philosophy_mode: str = 'none', philosophy_params=None,
                agent_kwargs=None, planner: str = None, planner_params=None):
    return run_episode_with_state(steps, cfg, learn, philosophy_mode, philosophy_params, agent_kwargs, planner, planner_params)[0]

def run_episode_with_state(steps=300, cfg: PGConfig = PGConfig(), learn=True,
                           philosophy_mode: str = 'none', philosophy_params=None,
                           agent_kwargs=None, planner: str = None, planner_params=None):
    env = PublicGoodsGrid(cfg)
    obs = env.reset()
    agents = _make_agents(env, philosophy_mode, philosophy_params, agent_kwargs)
    planner_params = planner_params or {}; other_policy = planner_params.get('other_policy', 'stay')
    logs = {'F':[], 'valence':[], 'harmony':[], 'resource_mean':[], 'public_pool':[], 'ext_net':[], 'ext_harm':[], 'ext_benefit':[],
            'D_B':[], 'C_Stoic':[], 'Resp_KL':[]}
    current_mean_field = np.ones(cfg.act_dim) / cfg.act_dim

    for t in range(steps):
        acts = []
        for i in range(cfg.n_agents):
            neigh_idx = neighbor_indices(env, i)
            neigh_obs = [obs[j] for j in neigh_idx]
            sim = None
            if planner == 'cem':
                sim = make_simulator(env, i, other_policy, current_mean_field)
            a, _ = agents[i].plan(obs[i], neigh_obs, simulate=sim, planner=planner, planner_params=planner_params, agent_index=i)
            acts.append(a)
        
        counts = np.bincount(acts, minlength=cfg.act_dim).astype(np.float32)
        step_dist = counts / counts.sum()
        current_mean_field = 0.7 * current_mean_field + 0.3 * step_dist

        obs_next, rewards, done, info = env.step(acts)
        if learn:
            for i in range(cfg.n_agents):
                batch = np.stack([obs[i], obs_next[i]], axis=0)
                agents[i].update(batch)
        F_mean = np.mean([a.prev_F if a.prev_F is not None else 0.0 for a in agents])
        val_mean = np.mean([a.valence for a in agents])
        harm_vals = []
        for i in range(cfg.n_agents):
            vi = torch.tensor(obs[i]).float()
            nidx = neighbor_indices(env, i)
            for j in nidx:
                vj = torch.tensor(obs[j]).float()
                cs = torch.nn.functional.cosine_similarity(vi, vj, dim=0).item()
                harm_vals.append(cs)
        harmony = float(np.mean(harm_vals)) if len(harm_vals)>0 else 0.0
        logs['F'].append(F_mean); logs['valence'].append(val_mean); logs['harmony'].append(harmony)
        logs['resource_mean'].append(info['resource_mean']); logs['public_pool'].append(info['public_pool'])
        D_B_vals, C_Stoic_vals, Resp_vals = [], [], []
        for a in agents:
            if 'D_B' in a.last_metrics: D_B_vals.append(a.last_metrics['D_B'])
            if 'C_Stoic' in a.last_metrics: C_Stoic_vals.append(a.last_metrics['C_Stoic'])
            if 'Resp_KL' in a.last_metrics: Resp_vals.append(a.last_metrics['Resp_KL'])
        logs['D_B'].append(float(np.mean(D_B_vals)) if len(D_B_vals)>0 else 0.0)
        logs['C_Stoic'].append(float(np.mean(C_Stoic_vals)) if len(C_Stoic_vals)>0 else 0.0)
        logs['Resp_KL'].append(float(np.mean(Resp_vals)) if len(Resp_vals)>0 else 0.0)
        if 'ext_net' in info:
            en = info['ext_net']; eh = info['ext_harm']; eb = info['ext_benefit']
            logs['ext_net'].append(float(en.mean()))
            logs['ext_harm'].append(float(eh.mean()))
            logs['ext_benefit'].append(float(eb.mean()))
        else:
            logs['ext_net'].append(0.0); logs['ext_harm'].append(0.0); logs['ext_benefit'].append(0.0)
        obs = obs_next
        if done: break
    final_state = {'grid': env.grid.copy(), 'pos': env.pos.copy(), 'store': env.store.copy(), 'public_pool': env.public_pool}
    return logs, final_state

if __name__ == '__main__':
    cfg = PGConfig(size=12, n_agents=6, seed=42)
    logs = run_episode(steps=100, cfg=cfg, learn=True, planner='cem', 
                       planner_params={'H':5, 'N':32, 'iters':3, 'elite_frac':0.2, 'other_policy':'mean_field'})
    print({k: float(np.mean(v)) for k,v in logs.items()})
