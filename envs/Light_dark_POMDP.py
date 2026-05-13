import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import matplotlib.pyplot as plt

@dataclass
class LightDarkConfig:
    # World
    world_radius: float = 10.0
    dt: float = 0.2
    max_steps: int = 300

    # Inertial dynamics (Env-2)
    alpha: float = 0.98          # velocity decay (drag)
    beta: float = 1.0            # accel -> velocity gain
    a_max: float = 0.5           # action bound (acceleration)
    v_max: float = 2.0           # optional velocity clamp (keeps things stable)

    # Episode-level drift (hidden context)
    sigma_c: float = 0.6         # std of c ~ N(0, sigma_c^2 I)
    sigma_eta: float = 0.01      # process noise on velocity

    # Goal & termination
    goal: Tuple[float, float] = (7.0, 7.0)
    goal_radius: float = 0.5

    # Observation noise (position)
    sigma_dark: float = 2.0
    sigma_light: float = 0.05

    # Band geometry
    band_center: Tuple[float, float] = (0.0, 0.0)
    band_angle_deg: float = 90.0   # 90° = vertical strip
    band_width: float = 2.0

    # Band-only cue noise (Δv)
    sigma_cue_light: float = 0.03  # low noise in band
    sigma_cue_dark: float = 1e3    # effectively masked out when dark
    hard_mask_cue_in_dark: bool = True  # if True, set cue=(0,0) when dark

    # Observation composition
    include_goal_in_obs: bool = True
    noisy_goal_obs: bool = False

    # Episode randomization
    randomize_start: bool = True
    randomize_goal: bool = True
    min_start_goal_dist: float = 6.0
    start_outside_band_prob: float = 0.8
    require_opposite_band_side: bool = False

    # Episode-level drift (hidden context)
    drift_mean: Tuple[float, float] = (0.0, 0.0)  # mean of c
    sigma_c: float = 0.6                           # std of c (if not fixed)
    fixed_c: Optional[Tuple[float, float]] = None  # if set, use this exact c every episode

    # Rewards
    step_cost: float = -0.01
    distance_scale: float = 0.0
    success_reward: float = 1.0

    # Seeding
    seed: Optional[int] = None


class LightDarkNavigationEnv(Env):
    """
    Env-2: Inertial dynamics + hidden per-episode drift + band-only Δv cue.

    Hidden state: x∈R^2 (position), v∈R^2 (velocity), c∈R^2 (drift, fixed per episode)
    Action (accel): a∈[-a_max, a_max]^2
    Dynamics: v_{t+1}=α v_t + β a_t + c + η ; x_{t+1}=x_t + v_{t+1}·dt
    Observation: [ noisy_pos(2), cue Δv(2), cue_mask(1) ] (+ optional goal(2))
      - cue is reliable (low-noise) only if start-of-step position was in the light band
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: Optional[LightDarkConfig] = None, render_mode: Optional[str] = None):
        self.cfg = config or LightDarkConfig()
        self.render_mode = render_mode
        self.rng = np.random.default_rng(self.cfg.seed)

        # Actions are accelerations
        self.action_space = Box(
            low=-self.cfg.a_max, high=self.cfg.a_max, shape=(2,), dtype=np.float32
        )

        # Observation: pos(2) + cue Δv(2) + mask(1) [+ goal(2)?]
        base_obs_dim = 2 + 2 + 1
        obs_dim = base_obs_dim + (2 if self.cfg.include_goal_in_obs else 0)
        high = np.full((obs_dim,), np.finfo(np.float32).max)
        self.observation_space = Box(low=-high, high=high, shape=(obs_dim,), dtype=np.float32)

        # Internals
        self._x = None  # true position (2,)
        self._v = None  # true velocity (2,)
        self._c = None  # per-episode drift (2,)
        self._steps = 0

        # For timing the band-only cue (belongs to o_{t+1})
        self._cue = np.zeros(2, dtype=np.float32)
        self._cue_mask = 0.0  # 1.0 if reliable (in band at step start)

        # Precompute band geometry
        theta = np.deg2rad(self.cfg.band_angle_deg)
        self._u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        self._n = np.array([-self._u[1], self._u[0]], dtype=np.float32)
        self._c_band = np.array(self.cfg.band_center, dtype=np.float32)

        # Matplotlib objects
        self._fig = None
        self._ax = None

    # ---------------------- Gymnasium API ----------------------
    def seed(self, seed: Optional[int] = None):
        self.cfg.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.seed(seed)   
        self._steps = 0     
        R = self.cfg.world_radius

        # Read optional per-episode overrides for start/goal from options (else None).
        opt_start = None if options is None else options.get("start", None)
        opt_goal = None if options is None else options.get("goal", None)

        # If caller provided a start, use it.
        # Else if randomizing: sample uniformly in the box until it passes a constraint
        # with probability start_outside_band_prob, force the start to be outside 
        # the light band (resample if it lands in the band).
        if opt_start is not None:
            start = np.asarray(opt_start, dtype=np.float32)
        elif self.cfg.randomize_start:
            while True:
                start = self.rng.uniform(low=-R, high=R, size=(2,)).astype(np.float32)
                if self.rng.random() < self.cfg.start_outside_band_prob and self._in_light(start):
                    continue
                break
        else:
            start = np.zeros(2, dtype=np.float32)  # If not randomized and no override, default to origin.

        # Sample goal
        if opt_goal is not None:
            goal = np.asarray(opt_goal, dtype=np.float32) # If caller provided a goal, use it.
        elif self.cfg.randomize_goal:
            while True:
                goal = self.rng.uniform(low=-R, high=R, size=(2,)).astype(np.float32)
                if np.linalg.norm(goal - start) < self.cfg.min_start_goal_dist:  # Else if randomizing: sample until the goal is far enough from start (at least min_start_goal_dist).
                    continue
                # If required, also enforce start and goal on opposite sides of the band centerline 
                # (dot products along the band normal must have opposite signs).
                if self.cfg.require_opposite_band_side:
                    sx = float(np.dot(start - self._c_band, self._n))
                    gx = float(np.dot(goal - self._c_band, self._n))
                    if sx * gx >= 0:
                        continue
                break
        # If not randomized and no override: use configured goal. If it’s too close to start,
        # nudge it along the band direction _u to respect the min distance.
        else:
            goal = np.array(self.cfg.goal, dtype=np.float32)
            if np.linalg.norm(goal - start) < self.cfg.min_start_goal_dist:
                goal = start + (self.cfg.min_start_goal_dist + 1.0) * self._u

        # Commit hidden state
        self._x = start.astype(np.float32)
        self._v = np.zeros(2, dtype=np.float32)
        # Per-episode drift (order of precedence: options["c"] > fixed_c > random)
        if options is not None and "c" in options:
            c = np.asarray(options["c"], dtype=np.float32)
        elif self.cfg.fixed_c is not None:
            c = np.asarray(self.cfg.fixed_c, dtype=np.float32)
        else:
            mean = np.asarray(self.cfg.drift_mean, dtype=np.float32)
            c = mean + self.rng.normal(0.0, self.cfg.sigma_c, size=(2,)).astype(np.float32)
        self._c = c


        # Store the chosen goal back into the config (so rendering/observation sees it).
        self.cfg.goal = (float(goal[0]), float(goal[1]))

        # At reset, no previous action → cue not available
        self._cue = np.zeros(2, dtype=np.float32)
        self._cue_mask = 0.0

        # Create the observation at time 0: noisy position + (zero) cue + mask (+ goal if enabled).
        obs = self._observe(self._x, self._cue, self._cue_mask)
        info = {
            "true_state": np.concatenate([self._x, self._v, self._c]).astype(np.float32),
            "sigma_pos": self._sigma_at(self._x),
            "start": self._x.copy(),
            "goal": goal.astype(np.float32),
            "cue_mask": self._cue_mask,
        }
        return obs, info

    def step(self, action: np.ndarray):
        self._steps += 1

        # Entry-point band rule: decide cue reliability based on x_t
        in_band_start = self._in_light(self._x)
        cue_mask_next = 1.0 if in_band_start else 0.0

        # Clip action (acceleration)
        a = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)

        # Process noise on velocity
        eta = self.rng.normal(0.0, self.cfg.sigma_eta, size=(2,)).astype(np.float32)

        # Inertial update
        v_next = self.cfg.alpha * self._v + self.cfg.beta * a + self._c + eta
        # Optional velocity clamp (not displacement!)
        v_norm = np.linalg.norm(v_next)
        if v_norm > self.cfg.v_max:
            v_next = (v_next / (v_norm + 1e-8)) * self.cfg.v_max

        x_next = self._x + v_next * self.cfg.dt

        # Keep inside bounds
        R = self.cfg.world_radius
        x_next = np.clip(x_next, -R, R)

        # True Δv for the cue
        delta_v = (v_next - self._v).astype(np.float32)

        # Build the cue observed at o_{t+1}
        if cue_mask_next >= 0.5:
            # reliable cue in band
            zeta = self.rng.normal(0.0, self.cfg.sigma_cue_light, size=(2,)).astype(np.float32)
            cue_next = (delta_v + zeta).astype(np.float32)
        else:
            if self.cfg.hard_mask_cue_in_dark:
                cue_next = np.zeros(2, dtype=np.float32)
            else:
                zeta = self.rng.normal(0.0, self.cfg.sigma_cue_dark, size=(2,)).astype(np.float32)
                cue_next = (delta_v + zeta).astype(np.float32)

        # Commit state
        self._x, self._v = x_next, v_next
        self._cue, self._cue_mask = cue_next, cue_mask_next

        # Observation at t+1
        obs = self._observe(self._x, self._cue, self._cue_mask)

        # Reward (unchanged for now)
        dist = float(np.linalg.norm(self._x - np.array(self.cfg.goal, dtype=np.float32)))
        reward = self.cfg.step_cost + self.cfg.distance_scale * dist
        terminated = dist <= self.cfg.goal_radius
        if terminated:
            reward += self.cfg.success_reward
        truncated = self._steps >= self.cfg.max_steps

        info = {
            "true_state": np.concatenate([self._x, self._v, self._c]).astype(np.float32),
            "sigma_pos": self._sigma_at(self._x),
            "distance": dist,
            "in_light": bool(self._in_light(self._x)),
            "cue_mask": float(self._cue_mask),
            "delta_v_true": delta_v.copy(),
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    # ---------------------- Observation model ----------------------
    #return the observation noise std to use at position x, depending on whether
    #  x is inside the light band (low noise) or outside (dark, high noise).
    # how? how far is x from the band centerline? If within half-width,
    #  use light noise; otherwise dark noise
    def _sigma_at(self, x: np.ndarray) -> float:
        d = abs(float(np.dot(x - self._c_band, self._n)))
        return self.cfg.sigma_light if d <= (self.cfg.band_width / 2.0) else self.cfg.sigma_dark

    def _in_light(self, x: np.ndarray) -> bool:
        return self._sigma_at(x) == self.cfg.sigma_light

    def _observe(self, x: np.ndarray, cue: np.ndarray, cue_mask: float) -> np.ndarray:
        # Noisy position
        sigma = self._sigma_at(x)
        noise = self.rng.normal(0.0, sigma, size=(2,)).astype(np.float32)
        y_pos = (x + noise).astype(np.float32)

        parts = [y_pos, cue.astype(np.float32), np.array([cue_mask], dtype=np.float32)]
        if self.cfg.include_goal_in_obs:
            g = np.array(self.cfg.goal, dtype=np.float32)
            if self.cfg.noisy_goal_obs:
                sigma_g = self._sigma_at(g)
                g = (g + self.rng.normal(0.0, sigma_g, size=(2,))).astype(np.float32)
            parts.append(g.astype(np.float32))
        return np.concatenate(parts).astype(np.float32)

    # ---------------------- Rendering (unchanged) ----------------------
    def render(self):
        if self.render_mode is None:
            return None
        if plt is None:
            raise RuntimeError("matplotlib is required for rendering")

        if self._fig is None:
            if self.render_mode == "human":
                try:
                    plt.ion()
                except Exception:
                    pass
            self._fig, self._ax = plt.subplots(figsize=(5, 5))

        ax = self._ax
        ax.clear()
        ax.set_aspect('equal')
        ax.set_xlim(-self.cfg.world_radius, self.cfg.world_radius)
        ax.set_ylim(-self.cfg.world_radius, self.cfg.world_radius)
        ax.set_title("Light-Dark Navigation (Env-2)")
        ax.set_facecolor('black')

        R = self.cfg.world_radius
        span = R * 2.0
        line_pts = np.stack([self._c_band - span * self._u, self._c_band + span * self._u], axis=0)
        half_w = self.cfg.band_width * 0.5
        p1 = line_pts[0] + (-half_w) * self._n
        p2 = line_pts[1] + (-half_w) * self._n
        p3 = line_pts[1] + (half_w) * self._n
        p4 = line_pts[0] + (half_w) * self._n
        from matplotlib.patches import Polygon
        band_poly = Polygon(np.vstack([p1, p2, p3, p4]), closed=True,
                            facecolor='white', edgecolor='white', alpha=1.0, zorder=0)
        ax.add_patch(band_poly)

        g = np.array(self.cfg.goal, dtype=np.float32)
        ax.scatter([g[0]], [g[1]], marker='*', s=200, label='Goal')
        if self._x is not None:
            ax.scatter([self._x[0]], [self._x[1]], s=60, label='Agent')
            sig = self._sigma_at(self._x)
            circ = plt.Circle((self._x[0], self._x[1]), sig, fill=False, alpha=0.5)
            ax.add_patch(circ)

        ax.legend(loc='upper left', facecolor='white')
        ax.grid(True, alpha=0.2)

        self._fig.canvas.draw()
        if self.render_mode == "human":
            plt.pause(0.5)
            return None
        elif self.render_mode == "rgb_array":
            w, h = self._fig.canvas.get_width_height()
            img = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            return img.reshape((h, w, 3))

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig, self._ax = None, None

# Factory
def make_env(*, render_mode: Optional[str] = None, **kwargs) -> LightDarkNavigationEnv:
    cfg = LightDarkConfig(**kwargs)
    return LightDarkNavigationEnv(config=cfg, render_mode=render_mode)

if __name__ == "__main__":
    # Make Env-2 (inertial + drift + band-only Δv cue)
    env = make_env(
        render_mode="human",
        world_radius=10.0,
        dt=0.2,
        max_steps=300,
        alpha=0.98,
        beta=1.0,
        a_max=20.0,
        v_max=50.0,
        fixed_c=(0.8, 0.2),
        sigma_c=0.0,
        sigma_eta=0.01,
        # light–dark sensing
        band_angle_deg=90.0,    # vertical white strip
        band_center=(-8.0 + 2.0/2, 0.0),  # left side
        band_width=2.0,
        sigma_dark=2.0,
        sigma_light=0.05,
        # obs composition
        include_goal_in_obs=True,
        noisy_goal_obs=False,
        # episode randomization
        randomize_start=True,
        randomize_goal=True,
        min_start_goal_dist=6.0,
        start_outside_band_prob=0.9,
        require_opposite_band_side=False,
    )

    obs, info = env.reset()
    print("obs shape:", obs.shape, "| first obs:", obs)

    done = False
    while not done:
        #a = env.action_space.sample()          # random acceleration
        a = np.array([0.0, 0.0], dtype=np.float32)  # zero acceleration
        obs, r, terminated, truncated, info = env.step(a)
        env.render()                            # live window
        done = terminated or truncated

    env.close()