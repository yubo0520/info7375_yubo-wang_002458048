import copy
import io
import contextlib
import numpy as np
import gymnasium as gym
import minigrid  # noqa: F401 -- registers BabyAI envs


# paper uses this as "Goto" in Table 1
ENV_NAME = "BabyAI-GoTo-v0"

# each room is ~7 cells wide (6 + 1 wall), 22x22 grid = 3x3 rooms
ROOM_SIZE = 7
MAX_STEPS = 100


def room_of(x, y):
    # map grid cell to room index
    return (int(x) // ROOM_SIZE, int(y) // ROOM_SIZE)


def make_env():

    return gym.make(ENV_NAME, render_mode=None, max_steps=MAX_STEPS)


class GoToEnv:
    # thin wrapper: tracks rooms visited per episode, supports state save/restore
    def __init__(self):
        self.env = make_env()
        self._rooms = set()
        self._room_seq = []
        self._step_count = 0

    def reset(self, seed=None):
        self._rooms = set()
        self._room_seq = []
        self._step_count = 0
        # Keep training logs clean and avoid excessive I/O overhead.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if seed is not None:
                obs, info = self.env.reset(seed=seed)
            else:
                obs, info = self.env.reset()
        ax, ay = self.env.unwrapped.agent_pos
        r = room_of(ax, ay)
        self._rooms.add(r)
        self._room_seq.append(r)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        ax, ay = self.env.unwrapped.agent_pos
        r = room_of(ax, ay)
        self._rooms.add(r)
        if not self._room_seq or self._room_seq[-1] != r:
            self._room_seq.append(r)
        done = terminated or truncated

        # rescale reward -- paper uses 1 - 0.5*(t/H) instead of default 1 - 0.9*(t/H)
        if reward > 0:
            reward = 1.0 - 0.5 * (self._step_count / MAX_STEPS)

        return obs, reward, done, info

    def rooms_visited(self):
        return frozenset(self._rooms)

    def room_sequence(self):
        return tuple(self._room_seq)

    def get_obs(self):
        # get current obs without stepping (used after restore)
        return self.env.unwrapped.gen_obs()

    def save_state(self):
        # deepcopy the underlying unwrapped env for vine sampling resets
        return (
            copy.deepcopy(self.env.unwrapped),
            frozenset(self._rooms),
            tuple(self._room_seq),
            self._step_count,
        )

    def restore_state(self, state):
        env_snap, rooms, room_seq, step_count = state
        self.env.unwrapped.__dict__.update(copy.deepcopy(env_snap).__dict__)
        self._rooms = set(rooms)
        self._room_seq = list(room_seq)
        self._step_count = step_count

    @property
    def n_actions(self):
        return self.env.action_space.n


def obs_to_arrays(obs):
    # split obs dict into (image, direction, mission_str)
    img = obs["image"].astype(np.float32) / 10.0   # normalize to ~[0,1]
    direction = int(obs["direction"])
    mission = obs["mission"]
    return img, direction, mission


if __name__ == "__main__":
    env = GoToEnv()
    obs = env.reset(seed=42)
    img, d, m = obs_to_arrays(obs)
    print(f"image shape: {img.shape}")
    print(f"direction: {d}, mission: '{m}'")
    print(f"n_actions: {env.n_actions}")

    state = env.save_state()
    obs2, r, done, _ = env.step(2)
    print(f"after step: rooms={env.rooms_visited()}, r={r:.3f}")

    env.restore_state(state)
    print(f"after restore: rooms={env.rooms_visited()}, step={env._step_count}")
