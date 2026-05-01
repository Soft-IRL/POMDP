import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
#import torch
import matplotlib.pyplot as plt



@dataclass
class LightDarkConfig:
    # World
    world_radius: float = 10.0  # square box [-R, R]^2
    dt: float = 1.0             # one action = one displacement unit
    max_speed: float = 1.0      # max |action| per axis
    max_steps: int = 200

    # Goal & termination
    goal: Tuple[float, float] = (7.0, 7.0)
    goal_radius: float = 0.5

    # Observation noise (position-only by default)
    sigma_dark: float = 2.0
    sigma_light: float = 0.1

    # "Light band" geometry: infinite line with finite thickness
    band_center: Tuple[float, float] = (0.0, 0.0)
    band_angle_deg: float = 0.0  # 0° = horizontal band (thin along y)
    band_width: float = 2.0

    # Observation composition
    include_goal_in_obs: bool = True   # append goal (possibly noisy) to observation
    noisy_goal_obs: bool = False       # if True, goal observation gets same band noise model

    # Episode randomization
    randomize_start: bool = True
    randomize_goal: bool = True
    min_start_goal_dist: float = 6.0
    start_outside_band_prob: float = 0.8  # probability to force start outside the band
    require_opposite_band_side: bool = False  # if True, put start and goal on opposite sides of band

    # Rewards
    step_cost: float = -0.01
    distance_scale: float = 0.0       # additional shaping: distance to goal * scale
    success_reward: float = 1.0

    # Seeding
    seed: Optional[int] = None


class LightDarkNavigationEnv(Env):
    """
    2D navigation with position observations that are very noisy in the "dark" and
    precise within a thin "light band". Ideal simple POMDP for testing information-seeking.

    State (hidden): agent position x ∈ [-R, R]^2 (float32)
    Action: delta position a ∈ [-max_speed, max_speed]^2
    Observation: concat([noisy_position, goal_obs?]) where noise σ(x) is σ_light within band and σ_dark outside.

    Termination when ||x - g|| ≤ goal_radius or after max_steps (truncation).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: Optional[LightDarkConfig] = None, render_mode: Optional[str] = None):
        self.cfg = config or LightDarkConfig()
        self.render_mode = render_mode
        self.rng = np.random.default_rng(self.cfg.seed)

        # Continuous action: delta position per step
        self.action_space = Box(low=-self.cfg.max_speed, high=self.cfg.max_speed, shape=(2,), dtype=np.float32)

        # Observation: noisy position (2,) plus optional goal (2,)
        obs_dim = 2 + (2 if self.cfg.include_goal_in_obs else 0)
        high = np.full((obs_dim,), np.finfo(np.float32).max)
        self.observation_space = Box(low=-high, high=high, shape=(obs_dim,), dtype=np.float32)

        # Internals
        self._x = None  # true position
        self._steps = 0

        # Precompute band geometry
        theta = np.deg2rad(self.cfg.band_angle_deg)
        self._u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)  # direction along band centerline
        self._n = np.array([-self._u[1], self._u[0]], dtype=np.float32)       # normal to the band
        self._c = np.array(self.cfg.band_center, dtype=np.float32)

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

        # Allow overrides via options
        opt_start = None if options is None else options.get("start", None)
        opt_goal = None if options is None else options.get("goal", None)

        # Sample start
        if opt_start is not None:
            start = np.asarray(opt_start, dtype=np.float32)
        elif self.cfg.randomize_start:
            fixed_goal = np.array(self.cfg.goal, dtype=np.float32)

            while True:
                start = self.rng.uniform(low=-R, high=R, size=(2,)).astype(np.float32)

                # With prob p, enforce start outside the band
                if self.rng.random() < self.cfg.start_outside_band_prob and self._in_light(start):
                    continue

                # If goal is fixed (randomize_goal=False and no override), enforce min distance by resampling start
                if (opt_goal is None) and (not self.cfg.randomize_goal):
                    if np.linalg.norm(fixed_goal - start) < self.cfg.min_start_goal_dist:
                        continue
                break
        else:
            # Keep previous position or reset to origin if none
            start = np.zeros(2, dtype=np.float32)

        # Sample goal
        if opt_goal is not None:
            goal = np.asarray(opt_goal, dtype=np.float32)
        elif self.cfg.randomize_goal:
            while True:
                goal = self.rng.uniform(low=-R, high=R, size=(2,)).astype(np.float32)
                if np.linalg.norm(goal - start) < self.cfg.min_start_goal_dist:
                    continue
                if self.cfg.require_opposite_band_side:
                    # Opposite sides if (n·(x-c))*(n·(g-c)) < 0
                    sx = float(np.dot(start - self._c, self._n))
                    gx = float(np.dot(goal - self._c, self._n))
                    if sx * gx >= 0:
                        continue
                break
        else:
            goal = np.array(self.cfg.goal, dtype=np.float32)

        # Commit
        self._x = start.astype(np.float32)
        self.cfg.goal = (float(goal[0]), float(goal[1]))

        obs = self._observe(self._x)
        info = {
            "true_state": self._x.copy(),
            "sigma": self._sigma_at(self._x),
            "start": self._x.copy(),
            "goal": goal.astype(np.float32),
        }
        return obs, info

    def step(self, action: np.ndarray):
        self._steps += 1
        a = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)

        # Transition
        x_next = self._x + a * self.cfg.dt
        R = self.cfg.world_radius
        x_next = np.clip(x_next, -R, R)
        self._x = x_next

        # Observation
        obs = self._observe(self._x)

        # Rewards
        dist = float(np.linalg.norm(self._x - np.array(self.cfg.goal, dtype=np.float32)))
        reward = self.cfg.step_cost + self.cfg.distance_scale * dist

        terminated = dist <= self.cfg.goal_radius
        if terminated:
            reward += self.cfg.success_reward

        truncated = self._steps >= self.cfg.max_steps

        info = {
            "true_state": self._x.copy(),
            "sigma": self._sigma_at(self._x),
            "distance": dist,
            "in_light": bool(self._in_light(self._x)),
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    # ---------------------- Observation model ----------------------
    def _sigma_at(self, x: np.ndarray) -> float:
        # Distance to band centerline = | (x - c)·n |
        d = abs(float(np.dot(x - self._c, self._n)))
        return self.cfg.sigma_light if d <= (self.cfg.band_width / 2.0) else self.cfg.sigma_dark

    def _in_light(self, x: np.ndarray) -> bool:
        return self._sigma_at(x) == self.cfg.sigma_light

    def _observe(self, x: np.ndarray) -> np.ndarray:
        sigma = self._sigma_at(x)
        noise = self.rng.normal(loc=0.0, scale=sigma, size=(2,)).astype(np.float32)
        y = (x + noise).astype(np.float32)
        if self.cfg.include_goal_in_obs:
            g = np.array(self.cfg.goal, dtype=np.float32)
            if self.cfg.noisy_goal_obs:
                sigma_g = self._sigma_at(g)
                g = (g + self.rng.normal(0.0, sigma_g, size=(2,))).astype(np.float32)
            return np.concatenate([y, g]).astype(np.float32)
        return y

    # ---------------------- Rendering ----------------------
    def render(self):
        """Minimal, reliable matplotlib rendering.
        - "human": draws and non-blocking pauses.
        - "rgb_array": returns an RGB frame (H,W,3).
        """
        if self.render_mode is None:
            return None
        if plt is None:
            raise RuntimeError("matplotlib is required for rendering")

        # Create figure/axes once
        if self._fig is None:
            if self.render_mode == "human":
                try:
                    plt.ion()  # interactive, non-blocking
                except Exception:
                    pass
            self._fig, self._ax = plt.subplots(figsize=(5, 5))

        ax = self._ax
        ax.clear()
        ax.set_aspect('equal')
        ax.set_xlim(-self.cfg.world_radius, self.cfg.world_radius)
        ax.set_ylim(-self.cfg.world_radius, self.cfg.world_radius)
        ax.set_title("Light-Dark Navigation")
        # Set high-uncertainty (dark) region as black background
        ax.set_facecolor('black')

        # Draw light band: filled white quad on top of black background
        R = self.cfg.world_radius
        # Endpoints of the centerline spanning beyond the view
        span = R * 2.0
        line_pts = np.stack([self._c - span * self._u, self._c + span * self._u], axis=0)
        # Quad corners = centerline ± (band_width/2) * normal
        half_w = self.cfg.band_width * 0.5
        p1 = line_pts[0] + (-half_w) * self._n
        p2 = line_pts[1] + (-half_w) * self._n
        p3 = line_pts[1] + (half_w) * self._n
        p4 = line_pts[0] + (half_w) * self._n
        from matplotlib.patches import Polygon
        band_poly = Polygon(np.vstack([p1, p2, p3, p4]), closed=True, facecolor='white', edgecolor='white', alpha=1.0, zorder=0)
        ax.add_patch(band_poly)

        # Optional: outline band edges
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle='--', linewidth=1, color='gray', zorder=1)
        ax.plot([p4[0], p3[0]], [p4[1], p3[1]], linestyle='--', linewidth=1, color='gray', zorder=1)

        # Draw goal and agent
        g = np.array(self.cfg.goal, dtype=np.float32)
        ax.scatter([g[0]], [g[1]], marker='*', s=200, label='Goal')
        if self._x is not None:
            ax.scatter([self._x[0]], [self._x[1]], s=60, label='Agent')
            sig = self._sigma_at(self._x)
            circ = plt.Circle((self._x[0], self._x[1]), sig, fill=False, alpha=0.5)
            ax.add_patch(circ)

        ax.legend(loc='upper left', facecolor='white')
        ax.grid(True, alpha=0.2)

        # Draw now
        self._fig.canvas.draw()

        if self.render_mode == "human":
            # Short pause updates the window without blocking
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

        if self._fig is not None:
            plt.close(self._fig)
            self._fig, self._ax = None, None


# Convenience factory for Gymnasium-style registration (optional)
def make_env(*, render_mode: Optional[str] = None, **kwargs) -> LightDarkNavigationEnv:
    """Factory compatible with gymnasium.make(..., render_mode=...).
    Any other kwargs are forwarded to LightDarkConfig.
    """
    cfg = LightDarkConfig(**kwargs)
    return LightDarkNavigationEnv(config=cfg, render_mode=render_mode)



if __name__ == "__main__":

    env = make_env(
    render_mode="human",
    world_radius=10.0,
    band_width=2.0,
    band_angle_deg=90.0,
    band_center=(-8.0 + 2.0/2, 0.0),  # i.e., (-9.0, 0.0)
    sigma_dark=0.0,
    sigma_light=0.0,
    include_goal_in_obs=True,
    randomize_start=False,
    randomize_goal=False,
    noisy_goal_obs=False,
)

    obs, info = env.reset()
    for _ in range(5):
        action = env.action_space.sample()
        print(action)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            break
    env.close()