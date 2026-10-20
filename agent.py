"""From-scratch on-policy SARSA agent for Gymnasium Taxi (discrete state/action)."""
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class SARSACfg:
    episodes: int = 20000
    alpha: float = 0.20
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    decay_fraction: float = 0.75
    max_steps: int = 200
    seed: int = 44


def epsilon_at(episode, cfg):
    progress = min(1.0, episode / max(1.0, cfg.episodes * cfg.decay_fraction))
    return cfg.epsilon_start + (cfg.epsilon_end - cfg.epsilon_start) * progress


def epsilon_greedy(q, state, epsilon, rng):
    """Random tie-breaking prevents the zero-initialized table favoring action 0."""
    if rng.random() < epsilon:
        return int(rng.integers(q.shape[1]))
    maxima = np.flatnonzero(q[state] == np.max(q[state]))
    return int(rng.choice(maxima))


def train_sarsa(env_factory, cfg: SARSACfg, verbose=True):
    """Learn SARSA's on-policy Q(s,a) table; time-limit truncated states bootstrap."""
    rng = np.random.default_rng(cfg.seed)
    env = env_factory()
    assert hasattr(env.observation_space, 'n') and hasattr(env.action_space, 'n'), 'Use discrete spaces'
    q = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)
    rewards, successes, steps_record, eps_record = [], [], [], []
    for ep in range(cfg.episodes):
        state, _ = env.reset(seed=cfg.seed + ep)
        eps = epsilon_at(ep, cfg)
        action = epsilon_greedy(q, state, eps, rng)
        reward_sum, success, steps = 0.0, False, 0
        for t in range(cfg.max_steps):
            state_next, reward, terminated, truncated, _ = env.step(action)
            # SARSA bootstraps on next *selected* action, except true terminal states.
            if terminated:
                target = reward
            else:
                next_action = epsilon_greedy(q, state_next, eps, rng)
                target = reward + cfg.gamma * q[state_next, next_action]
            q[state, action] += cfg.alpha * (target - q[state, action])
            reward_sum += reward
            steps = t + 1
            success = bool(terminated)
            if terminated or truncated:
                break
            state, action = state_next, next_action
        rewards.append(float(reward_sum))
        successes.append(int(success))
        steps_record.append(steps)
        eps_record.append(eps)
        if verbose and (ep + 1) % max(1, cfg.episodes // 5) == 0:
            sl = min(250, len(rewards))
            print(f'episode {ep+1:>6}/{cfg.episodes}: last-{sl} reward={np.mean(rewards[-sl:]):.2f}, success={np.mean(successes[-sl:])*100:.1f}%, eps={eps:.3f}')
    env.close()
    return q, {'rewards': rewards, 'successes': successes, 'steps': steps_record, 'epsilons': eps_record}


def evaluate(env_factory, q=None, episodes=100, seed=50000, max_steps=200, random_policy=False):
    rng = np.random.default_rng(seed + 711)
    env = env_factory()
    values, success, step_counts = [], [], []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        total, ended, n = 0.0, False, 0
        for t in range(max_steps):
            if random_policy:
                action = int(rng.integers(env.action_space.n))
            else:
                row = q[state]
                action = int(np.flatnonzero(row == np.max(row))[0])
            state, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            n = t + 1
            if terminated or truncated:
                ended = bool(terminated)
                break
        values.append(total)
        success.append(int(ended))
        step_counts.append(n)
    env.close()
    return {'mean_reward': float(np.mean(values)), 'success_rate': float(np.mean(success)),
            'mean_steps': float(np.mean(step_counts)), 'reward_std': float(np.std(values)),
            'episodes': episodes}


def successful_rollout(env_factory, q, search_seeds=500, max_steps=200):
    """Return a genuine successful greedy rollout for visual inspection."""
    env = env_factory()
    for seed in range(60000, 60000 + search_seeds):
        state, _ = env.reset(seed=seed)
        actions, rewards = [], []
        done = False
        for t in range(max_steps):
            a = int(np.argmax(q[state]))
            new_state, r, term, trunc, _ = env.step(a)
            actions.append(a)
            rewards.append(r)
            state = new_state
            if term or trunc:
                done = bool(term)
                break
        if done:
            env.close()
            return {'seed': seed, 'actions': actions, 'total_reward': sum(rewards), 'steps': len(actions), 'success': True}
    env.close()
    return None
