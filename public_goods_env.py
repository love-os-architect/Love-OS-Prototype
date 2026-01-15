import numpy as np
from dataclasses import dataclass

@dataclass
class PGConfig:
    size: int = 12                  # Grid size
    n_agents: int = 6               # Number of agents
    seed: int = 0
    obs_radius: int = 2             # Observation radius (Partial observation)
    base_regrow_p: float = 0.04     # Regrowth probability (per cell)
    base_regrow_amount: float = 0.08
    public_multiplier: float = 0.6  # Impact of public goods on regrowth
    overharvest_penalty: float = 0.5# Local regrowth multiplier upon overharvesting (<1 weakens it)
    act_dim: int = 7                # 0 stay, 1 up, 2 down, 3 left, 4 right, 5 harvest, 6 donate
    donate_unit: float = 0.2        # Donation unit (from own store to public goods)
    harvest_cap: float = 0.5        # Max harvest per step
    max_cell: float = 1.0           # Max cell resource
    # --- Externality parameters ---
    ext_eta_reg: float = 1.0        # Externality weight for local regrowth scale reduction
    ext_k_public: float = 1.0       # Positive externality weight for public goods donation
    ext_crowd_cost: float = 0.0     # Externality cost of movement crowding (unused)

class PublicGoodsGrid:
    """
    Public Goods Grid Environment (Partial Observation)
    - Resources in each cell [0, 1]. Probabilistic regrowth every step. 
    - Has a public goods pool (public_pool) that amplifies regrowth amount.
    - Harvest has externalities: Continued overharvesting temporarily reduces that cell's regrowth multiplier.
    - Actions: stay/move4/harvest/donate (donate to public goods)
    Observation vector: Neighborhood patch + Own store + Neighborhood average resource + Public goods level
    """
    def __init__(self, cfg: PGConfig = PGConfig()):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.size = cfg.size
        self.n_agents = cfg.n_agents
        self.obs_radius = cfg.obs_radius
        self.act_dim = cfg.act_dim
        self.reset()

    def reset(self):
        # Resource grid and local regrowth multipliers
        self.grid = self.rng.uniform(0.0, self.cfg.max_cell, size=(self.size, self.size)).astype(np.float32)
        self.local_regrow_scale = np.ones((self.size, self.size), dtype=np.float32)
        # Agent positions
        self.pos = self.rng.integers(0, self.size, size=(self.n_agents, 2))
        # Resources held by each agent
        self.store = np.zeros(self.n_agents, dtype=np.float32)
        # Public goods
        self.public_pool = 0.0
        self.t = 0
        return self._obs_all()

    # ----------------------- Observation -----------------------
    def _obs_i(self, i: int) -> np.ndarray:
        x, y = self.pos[i]
        R = self.obs_radius
        xs = np.arange(x-R, x+R+1)
        ys = np.arange(y-R, y+R+1)
        patch = np.zeros((2*R+1, 2*R+1), dtype=np.float32)
        for ix, xx in enumerate(xs):
            for iy, yy in enumerate(ys):
                if 0 <= xx < self.size and 0 <= yy < self.size:
                    patch[ix, iy] = self.grid[xx, yy]
        # Neighborhood mean resource (4-neighbors)
        neigh_coords = [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]
        neigh_vals = []
        for xx,yy in neigh_coords:
            if 0 <= xx < self.size and 0 <= yy < self.size:
                neigh_vals.append(self.grid[xx,yy])
        neigh_mean = float(np.mean(neigh_vals)) if len(neigh_vals)>0 else 0.0
        # Observation vector
        feat = np.array([
            self.store[i],                   # Own store
            neigh_mean,                      # Neighborhood mean resource
            self.public_pool / (self.size*self.size + 1e-6)  # Normalized public goods
        ], dtype=np.float32)
        return np.concatenate([patch.flatten(), feat], axis=0)

    def _obs_all(self):
        return [self._obs_i(i) for i in range(self.n_agents)]

    # ----------------------- Step -----------------------
    def step(self, acts):
        # Initialize externality ledger
        ext_harm = np.zeros(self.n_agents, dtype=np.float32)
        ext_benefit = np.zeros(self.n_agents, dtype=np.float32)
        # Movement
        for i, a in enumerate(acts):
            x, y = self.pos[i]
            if a == 1 and x > 0: self.pos[i,0] -= 1
            elif a == 2 and x < self.size-1: self.pos[i,0] += 1
            elif a == 3 and y > 0: self.pos[i,1] -= 1
            elif a == 4 and y < self.size-1: self.pos[i,1] += 1
        # Harvest / Donate
        for i, a in enumerate(acts):
            x, y = self.pos[i]
            if a == 5:  # harvest
                before_scale = float(self.local_regrow_scale[x,y])
                gain = min(self.cfg.harvest_cap, float(self.grid[x,y]))
                self.grid[x,y] -= gain
                self.store[i] += gain
                # Overharvest penalty (reduce local regrowth scale)
                self.local_regrow_scale[x,y] = max(0.1, self.local_regrow_scale[x,y] * self.cfg.overharvest_penalty)
                after_scale = float(self.local_regrow_scale[x,y])
                # Measured externality (negative): Resource removal + Regrowth scale reduction
                ext_harm[i] += gain + self.cfg.ext_eta_reg * max(0.0, before_scale - after_scale)
            elif a == 6:  # donate to public
                if self.store[i] > 0.05:
                    amt = min(self.cfg.donate_unit, float(self.store[i]))
                    self.store[i] -= amt
                    self.public_pool += amt
                    # Measured externality (positive): Public goods donation
                    ext_benefit[i] += self.cfg.ext_k_public * amt
        # Regrowth
        regrow_mask = (self.rng.random(self.grid.shape) < self.cfg.base_regrow_p).astype(np.float32)
        public_boost = self.cfg.public_multiplier * (self.public_pool / (self.size*self.size + 1e-6))
        regrow_amount = (self.cfg.base_regrow_amount + public_boost) * self.local_regrow_scale
        self.grid = np.clip(self.grid + regrow_mask * regrow_amount, 0.0, self.cfg.max_cell)
        # Natural recovery of local regrowth scale
        self.local_regrow_scale = np.clip(self.local_regrow_scale + 0.02, 0.1, 1.0)
        # Decay of public goods (maintenance cost)
        self.public_pool = max(0.0, self.public_pool * 0.995)
        # Observation
        obs = self._obs_all()
        # Reward (optional): Here, the increase in own store (reference value)
        rewards = self.store.copy()
        self.t += 1
        done = False
        info = {
            'public_pool': self.public_pool,
            'resource_mean': float(self.grid.mean()),
            'ext_harm': ext_harm,
            'ext_benefit': ext_benefit,
            'ext_net': ext_benefit - ext_harm,
        }
        return obs, rewards, done, info

    # Calculate observation dimension
    def obs_dim(self):
        R = self.obs_radius
        patch = (2*R+1) * (2*R+1)
        return patch + 3

    def clone(self, seed: int = None):
        """Deep copy of the environment (for simulation)"""
        new = PublicGoodsGrid(self.cfg)
        new.grid = self.grid.copy()
        new.local_regrow_scale = self.local_regrow_scale.copy()
        new.pos = self.pos.copy()
        new.store = self.store.copy()
        new.public_pool = float(self.public_pool)
        new.t = int(self.t)
        new.rng = np.random.default_rng(self.cfg.seed if seed is None else seed)
        return new
