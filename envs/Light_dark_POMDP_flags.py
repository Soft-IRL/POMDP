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
    c_max: float = None   # MUST satisfy c_max <= beta * a_max (here beta=1, a_max=0.5)
    sigma_c: float = 0.6         # std of c ~ N(0, sigma_c^2 I)
    sigma_eta: float = 0.01      # process noise on velocity

    # Goal & termination
    goal: Tuple[float, float] = (7.0, 7.0)
    goal_radius: float = 0.5
    
    # Strict stop condition (Env-2)
    v_tol: float = 0.2            # require ||v|| <= v_tol to count as success
    require_stop: bool = True     # if True, termination requires both position+stop

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

        # Episode-level tracking metrics
        self._t_first_band_entry = None   # int step index when first entering band (post-transition)
        self._band_visits = 0             # number of band *entries* this episode
        self._in_band_prev = False        # previous in_band flag (post-transition)
        self._reached_goal = False        # dist <= goal_radius (even if not stopped)
        self._success_strict = False      # strict stop success

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
        opt_goal  = None if options is None else options.get("goal", None)

        # ------------------------------
        # 1) Choose goal FIRST
        #   - If randomize_goal=False: goal is truly fixed (cfg.goal), never nudged.
        #   - If randomize_goal=True : sample uniformly in [-R,R]^2 (respect opposite-side constraint once start known).
        # ------------------------------
        if opt_goal is not None:
            goal = np.asarray(opt_goal, dtype=np.float32)
        elif self.cfg.randomize_goal:
            goal = None  # sample after start is known (for constraints involving start)
        else:
            goal = np.asarray(self.cfg.goal, dtype=np.float32)

        # Always ensure goal is in bounds (covers opt_goal and fixed goal cases)
        if goal is not None:
            goal = np.clip(goal, -R, R).astype(np.float32)

        # ------------------------------
        # 2) Choose start
        #   - If goal fixed and start randomized: resample start until far enough from fixed goal.
        #   - If goal randomized: sample start first (with outside-band constraint), then sample goal until far enough.
        # ------------------------------
        def _sample_start_once():
            s = self.rng.uniform(low=-R, high=R, size=(2,)).astype(np.float32)
            # With prob p, enforce start outside the band
            if self.rng.random() < self.cfg.start_outside_band_prob and self._in_light(s):
                return None
            return s

        # If caller provided a start, use it (clipped)
        if opt_start is not None:
            start = np.asarray(opt_start, dtype=np.float32)
            start = np.clip(start, -R, R).astype(np.float32)
        elif not self.cfg.randomize_start:
            start = np.zeros(2, dtype=np.float32)
        else:
            # randomized start
            max_tries = 10_000
            if goal is not None:
                # goal already fixed/overridden -> resample start until constraints satisfied
                for _ in range(max_tries):
                    s = _sample_start_once()
                    if s is None:
                        continue
                    if np.linalg.norm(goal - s) < self.cfg.min_start_goal_dist:
                        continue
                    if self.cfg.require_opposite_band_side:
                        sx = float(np.dot(s    - self._c_band, self._n))
                        gx = float(np.dot(goal - self._c_band, self._n))
                        if sx * gx >= 0:
                            continue
                    start = s
                    break
                else:
                    # fallback: accept last sampled start (or origin) to avoid infinite loop
                    start = s if s is not None else np.zeros(2, dtype=np.float32)
            else:
                # goal will be randomized later, so we only enforce start's own constraints here
                for _ in range(max_tries):
                    s = _sample_start_once()
                    if s is None:
                        continue
                    start = s
                    break
                else:
                    start = np.zeros(2, dtype=np.float32)

        # ------------------------------
        # 3) If goal is randomized, sample it now using the chosen start
        # ------------------------------
        if goal is None:
            max_tries = 10_000
            for _ in range(max_tries):
                g = self.rng.uniform(low=-R, high=R, size=(2,)).astype(np.float32)
                # ensure far enough from start
                if np.linalg.norm(g - start) < self.cfg.min_start_goal_dist:
                    continue
                # optional opposite-side constraint
                if self.cfg.require_opposite_band_side:
                    sx = float(np.dot(start - self._c_band, self._n))
                    gx = float(np.dot(g     - self._c_band, self._n))
                    if sx * gx >= 0:
                        continue
                goal = g
                break
            else:
                # fallback: keep sampling-free choice (clip) to avoid infinite loop
                goal = np.clip(g, -R, R).astype(np.float32)

            # final safety clip (should already be in bounds, but cheap and robust)
            goal = np.clip(goal, -R, R).astype(np.float32)

        # ------------------------------
        # 4) Commit hidden state
        # ------------------------------
        self._x = start.astype(np.float32)
        self._v = np.zeros(2, dtype=np.float32)

        # Per-episode drift (order of precedence: options["c"] > fixed_c > random feasible)
        if options is not None and "c" in options:
            c = np.asarray(options["c"], dtype=np.float32)

        elif self.cfg.fixed_c is not None:
            c = np.asarray(self.cfg.fixed_c, dtype=np.float32)

        else:
            mean = np.asarray(self.cfg.drift_mean, dtype=np.float32)

            # Feasibility bound: need |c_i| <= beta * a_max to be cancellable (component-wise).
            beta = float(getattr(self.cfg, "beta", 1.0))
            a_max = float(getattr(self.cfg, "a_max", np.max(np.abs(self.action_space.high))))
            if self.cfg.c_max is not None:
                c_max_cfg = float(getattr(self.cfg, "c_max", beta * a_max))
                c_max = min(c_max_cfg, beta * a_max)
            else:
                c_max = beta * a_max

            # Truncated Gaussian sampling: reject until within box [-c_max, c_max]^2
            sigma_c = float(getattr(self.cfg, "sigma_c", 0.0))
            max_tries = 10_000

            if sigma_c <= 0.0:
                c = mean.copy()
            else:
                for _ in range(max_tries):
                    c = mean + self.rng.normal(0.0, sigma_c, size=(2,)).astype(np.float32)
                    if np.all(np.abs(c) <= c_max):
                        break
                else:
                    # fallback: clip (should be extremely rare unless sigma_c is huge)
                    c = np.clip(c, -c_max, c_max).astype(np.float32)

        self._c = c.astype(np.float32)

        # Store the chosen goal back into the config (so rendering/observation sees it).
        self.cfg.goal = (float(goal[0]), float(goal[1]))

        # At reset, no previous action → cue not available
        self._cue = np.zeros(2, dtype=np.float32)
        self._cue_mask = 0.0

        # Reset episode metrics
        self._t_first_band_entry = None
        self._band_visits = 0
        self._in_band_prev = bool(self._in_light(self._x))
        if self._in_band_prev:
            self._band_visits = 1
            self._t_first_band_entry = 0
        self._reached_goal = False
        self._success_strict = False

        # Create the observation at time 0: noisy position + (zero) cue + mask (+ goal if enabled).
        obs = self._observe(self._x, self._cue, self._cue_mask)

        info = {
            "true_state": np.concatenate([self._x, self._v, self._c]).astype(np.float32),
            "sigma_pos": self._sigma_at(self._x),
            "start": self._x.copy(),
            "goal": goal.copy(),  # consistent with committed cfg.goal
            "c": self._c,
            "c_max": c_max,
            "cue_mask": self._cue_mask,
            "in_band": bool(self._in_light(self._x)),
            "t_first_band_entry": -1 if self._t_first_band_entry is None else int(self._t_first_band_entry),
            "band_visits": int(self._band_visits),
            "reached_goal": bool(self._reached_goal),
            "success_strict": bool(self._success_strict),
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

        # ---------------------- Episode metrics update ----------------------
        in_band_now = bool(self._in_light(self._x))  # after transition (state at t+1)
        if in_band_now and (not self._in_band_prev):
            self._band_visits += 1
        if self._t_first_band_entry is None and in_band_now:
            # first time we *arrive* in the band (post-transition)
            self._t_first_band_entry = int(self._steps)
        self._in_band_prev = in_band_now

        # ---------------------- Reward & termination ----------------------
        dist = float(np.linalg.norm(self._x - np.array(self.cfg.goal, dtype=np.float32)))
        speed = float(np.linalg.norm(self._v))

        reward = self.cfg.step_cost + self.cfg.distance_scale * dist

        reached_goal = dist <= self.cfg.goal_radius
        success_strict = bool(reached_goal and (speed <= float(self.cfg.v_tol)))

        # Termination logic:
        # - If require_stop=True: terminate only on strict success
        # - Else: terminate as soon as you reach the goal region (legacy behavior)
        if bool(self.cfg.require_stop):
            terminated = success_strict
        else:
            terminated = reached_goal

        if terminated:
            reward += self.cfg.success_reward

        truncated = self._steps >= self.cfg.max_steps

        # keep internal flags (for info + debugging)
        self._reached_goal = bool(reached_goal)
        self._success_strict = bool(success_strict)

        info = {
            "true_state": np.concatenate([self._x, self._v, self._c]).astype(np.float32),
            "sigma_pos": self._sigma_at(self._x),
            "distance": dist,
            "in_light": bool(self._in_light(self._x)),
            "cue_mask": float(self._cue_mask),
            "delta_v_true": delta_v.copy(),
            "in_band": bool(in_band_now),
            "t_first_band_entry": -1 if self._t_first_band_entry is None else int(self._t_first_band_entry),
            "band_visits": int(self._band_visits),
            "speed": float(speed),
            "reached_goal": bool(reached_goal),
            "success_strict": bool(success_strict),
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
        ax.set_title("Light-Dark Navigation (Env-2), c= " + str(self._c))
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
            plt.pause(0.001)
            return None
        elif self.render_mode == "rgb_array":
            buf = np.asarray(self._fig.canvas.buffer_rgba())
            return buf[..., :3].copy()

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
        c_max= None,  # MUST satisfy c_max <= beta * a_max (here beta=1, a_max=0.5)
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
        randomize_goal=False,
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