"""Local skeptic experiment runner for v2 multi-world brains.

Mirrors the notebook's architecture: multi-ball Catch, persistent hidden
state across balls, replay-on-stay, frozen + adaptive modes. 4 conditions
x 2 modes x N seeds.
"""
from __future__ import annotations

import json
import random as _python_random
import time
import numpy as np
from pathlib import Path


class Catch:
    def __init__(self, size=10, paddle_width=3, balls_per_episode=30, rng=None):
        self.size = size
        self.paddle_width = paddle_width
        self.balls_per_episode = balls_per_episode
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def _spawn_ball(self):
        self.ball_x = int(self.rng.integers(0, self.size))
        self.ball_y = 0

    def reset(self):
        self.paddle_x = self.size // 2
        self._spawn_ball()
        self.t = 0
        self.balls_completed = 0
        return self._obs()

    def _obs(self):
        return np.array([
            self.paddle_x / max(1, self.size - 1),
            self.ball_x / max(1, self.size - 1),
            self.ball_y / max(1, self.size - 1),
            (self.t % self.size) / max(1, self.size - 1),
        ], dtype=np.float64)

    def step(self, action: int):
        if action == 0:
            self.paddle_x = max(0, self.paddle_x - 1)
        elif action == 2:
            self.paddle_x = min(self.size - 1, self.paddle_x + 1)
        self.ball_y += 1
        self.t += 1
        landed = self.ball_y >= self.size - 1
        reward = 0.0
        if landed:
            half = self.paddle_width // 2
            reward = 1.0 if abs(self.ball_x - self.paddle_x) <= half else -1.0
            self.balls_completed += 1
            if self.balls_completed >= self.balls_per_episode:
                return self._obs(), reward, True
            self._spawn_ball()
        return self._obs(), reward, False


class TBI:
    def __init__(self, weights_in, weights_out, bias_h, bias_o,
                 attention_weights=None, attention_bias=None,
                 episodic_slots=None, episodic_age=None):
        self.weights_in = weights_in
        self.weights_out = weights_out
        self.bias_h = bias_h
        self.bias_o = bias_o
        self.attention_weights = attention_weights if (attention_weights is not None and attention_weights.size) else None
        self.attention_bias = attention_bias if (attention_bias is not None and attention_bias.size) else None
        self.episodic_slots = episodic_slots if (episodic_slots is not None and episodic_slots.size) else None
        self.episodic_age = episodic_age if (episodic_age is not None and episodic_age.size) else None
        self.input_size = weights_in.shape[1]
        self.hidden_size = weights_in.shape[0]
        self.output_size = weights_out.shape[0]
        self.hidden = np.zeros(self.hidden_size)
        self.last_inputs = np.zeros(self.input_size)

    @classmethod
    def from_checkpoint(cls, p):
        d = json.loads(Path(p).read_text())
        bd = d.get('brain') or d.get('brain_template') or d
        h = int(bd['hidden_size']); i = int(bd['input_size']); o = int(bd['output_size'])
        wi = np.array(bd['weights_in'], dtype=np.float64).reshape(h, i)
        wo = np.array(bd['weights_out'], dtype=np.float64).reshape(o, h)
        aw_raw = bd.get('attention_weights') or []
        ab_raw = bd.get('attention_bias') or []
        aw = np.array(aw_raw, dtype=np.float64).reshape(h, i) if len(aw_raw) == h * i else None
        ab = np.array(ab_raw, dtype=np.float64) if len(ab_raw) == i else None
        cap = int(bd.get('episodic_capacity') or 0)
        es_raw = bd.get('episodic_slots') or []
        ea_raw = bd.get('episodic_age') or []
        es = np.array(es_raw, dtype=np.float64).reshape(cap, h) if cap > 0 and len(es_raw) == cap * h else None
        ea = np.array(ea_raw, dtype=np.float64) if cap > 0 and len(ea_raw) == cap else None
        return cls(wi, wo, np.array(bd['bias_h']), np.array(bd['bias_o']), aw, ab, es, ea)

    @classmethod
    def random(cls, i, h, o, rng, with_episodic=False, capacity=0):
        wi = rng.normal(0, 0.45, (h, i))
        wo = rng.normal(0, 0.45, (o, h))
        bh = rng.normal(0, 0.08, h)
        bo = rng.normal(0, 0.08, o)
        aw = rng.normal(0, 0.15, (h, i))
        ab = rng.normal(3.0, 0.10, i)
        es = np.zeros((capacity, h)) if with_episodic and capacity > 0 else None
        ea = np.full(capacity, -1.0) if with_episodic and capacity > 0 else None
        return cls(wi, wo, bh, bo, aw, ab, es, ea)

    def reset(self):
        self.hidden = np.zeros(self.hidden_size)
        self.last_inputs = np.zeros(self.input_size)

    def _has_episodic(self):
        return self.episodic_slots is not None

    def _attend(self, x):
        if self.attention_weights is None:
            return x
        inv_h = 1.0 / np.sqrt(max(1, self.hidden_size))
        l = self.attention_bias + (self.hidden @ self.attention_weights) * inv_h
        np.clip(l, -30, 30, out=l)
        raw = 1.0 / (1.0 + np.exp(-l))
        budget = self.input_size * 0.95
        t = raw.sum()
        return x * (raw * (budget / t)) if t > budget and t > 0 else x * raw

    def _retrieve_episodes(self):
        if self.episodic_slots is None:
            return np.zeros(self.hidden_size)
        norm = np.sqrt(max(1.0, float(self.hidden_size)))
        scores = (self.episodic_slots @ self.hidden) / norm
        scores -= scores.max()
        weights = np.exp(scores)
        weights /= weights.sum() + 1e-9
        return weights @ self.episodic_slots

    def replay_episode(self, rng=None):
        if self.episodic_slots is None or self.episodic_slots.shape[0] < 2:
            return False
        valid = np.flatnonzero(self.episodic_age >= 0.0)
        if valid.size < 2:
            return False
        rng = rng or _python_random
        i_a, i_b = rng.sample(valid.tolist(), 2)
        replay_state = 0.5 * (self.episodic_slots[i_a] + self.episodic_slots[i_b])
        self.hidden = np.tanh(self.bias_h + 0.62 * replay_state)
        return True

    def forward(self, x):
        x = self._attend(np.asarray(x, dtype=np.float64))
        self.last_inputs = x
        inv = 1.0 / np.sqrt(max(1, self.input_size))
        nh = np.tanh(self.bias_h + 0.62 * self.hidden + (self.weights_in @ x) * inv)
        if self.episodic_slots is not None:
            self.hidden = nh
            retrieved = self._retrieve_episodes()
            nh = 0.82 * nh + 0.18 * retrieved
        self.hidden = nh
        return self.bias_o + (self.weights_out @ self.hidden) * (1.0 / np.sqrt(max(1, self.hidden_size)))

    def learn(self, action_index, valence, learning_rate=0.10, plasticity=0.5):
        if action_index < 0 or action_index >= self.output_size:
            return
        valence = max(-2.0, min(2.0, valence))
        lr = max(0.0, min(0.25, learning_rate)) * max(0.0, min(1.0, plasticity))
        self.weights_out[action_index] = np.clip(
            self.weights_out[action_index] + lr * valence * 0.035 * self.hidden, -4.0, 4.0,
        )
        self.bias_o[action_index] = max(-4.0, min(4.0, self.bias_o[action_index] + lr * valence * 0.015))
        repr_lr = lr * 0.020
        if repr_lr > 0.0 and self.last_inputs.size:
            hidden_gate = np.clip(self.hidden, -1.0, 1.0)
            delta = repr_lr * valence * np.outer(hidden_gate, self.last_inputs)
            self.weights_in = np.clip(self.weights_in + delta, -4.0, 4.0)


def shuffle_brain(brain, rng):
    def _shuf(arr):
        if arr is None:
            return None
        flat = arr.flatten().copy()
        rng.shuffle(flat)
        return flat.reshape(arr.shape)
    return TBI(
        _shuf(brain.weights_in), _shuf(brain.weights_out),
        _shuf(brain.bias_h), _shuf(brain.bias_o),
        _shuf(brain.attention_weights), _shuf(brain.attention_bias),
        _shuf(brain.episodic_slots), _shuf(brain.episodic_age),
    )


_REPLAY_RNG = _python_random.Random(0)


class CatchAdapter:
    def __init__(self, brain, rng):
        self.brain = brain
        self.rng = rng
        self.W_in = rng.normal(0, 0.30, (brain.input_size, 4))
        self.b_in = np.zeros(brain.input_size)
        self.W_out = rng.normal(0, 0.30, (3, brain.output_size))
        self.b_out = np.zeros(3)
        self.last_brain_outputs = None

    def reset(self):
        self.brain.reset()
        self.last_brain_outputs = None

    def logits(self, obs):
        bo = self.brain.forward(self.W_in @ obs + self.b_in)
        self.last_brain_outputs = bo
        return self.W_out @ bo + self.b_out

    def _maybe_replay(self, action):
        if action == 1 and self.brain._has_episodic():
            self.brain.replay_episode(_REPLAY_RNG)

    def greedy(self, obs):
        a = int(np.argmax(self.logits(obs)))
        self._maybe_replay(a)
        return a

    def stoch(self, obs):
        l = self.logits(obs)
        p = np.exp(l - l.max())
        p /= p.sum()
        a = int(self.rng.choice(3, p=p))
        self._maybe_replay(a)
        return a

    def observe_outcome(self, reward, adaptive_lr=0.0):
        if adaptive_lr <= 0.0 or reward == 0.0 or self.last_brain_outputs is None:
            return
        mg_action = int(np.argmax(self.last_brain_outputs))
        self.brain.learn(mg_action, valence=reward, learning_rate=adaptive_lr)

    def get(self):
        return self.W_in.copy(), self.W_out.copy(), self.b_in.copy(), self.b_out.copy()

    def set(self, w_in, w_out, b_in, b_out):
        self.W_in = w_in; self.W_out = w_out; self.b_in = b_in; self.b_out = b_out


class DirectPolicy:
    def __init__(self, rng):
        self.rng = rng
        self.W = rng.normal(0, 0.30, (3, 4))
        self.b = np.zeros(3)

    def reset(self):
        pass

    def greedy(self, obs):
        return int(np.argmax(self.W @ obs + self.b))

    def stoch(self, obs):
        l = self.W @ obs + self.b
        p = np.exp(l - l.max())
        p /= p.sum()
        return int(self.rng.choice(3, p=p))

    def get(self):
        return (self.W.copy(), self.b.copy())

    def set(self, W, b):
        self.W = W; self.b = b


def episode(adapter, env, greedy=False, adaptive_lr=0.0):
    adapter.reset()
    obs = env.reset()
    total = 0.0
    done = False
    while not done:
        action = adapter.greedy(obs) if greedy else adapter.stoch(obs)
        obs, r, done = env.step(action)
        total += r
        if hasattr(adapter, 'observe_outcome'):
            adapter.observe_outcome(r, adaptive_lr=adaptive_lr)
    return total


def evaluate(adapter, env, n=200, adaptive_lr=0.0):
    return float(np.mean([episode(adapter, env, greedy=True, adaptive_lr=adaptive_lr) for _ in range(n)]))


def train_es(adapter, env, gens=80, pop=24, sigma=0.20, lr=0.20, n_avg=3, adaptive_lr=0.0, seed=0):
    base = adapter.get()
    rng = np.random.default_rng(seed)
    for _ in range(gens):
        eps_all = []
        rewards = []
        for _ in range(pop):
            eps = tuple(rng.normal(0, sigma, p.shape) for p in base)
            adapter.set(*[b + e for b, e in zip(base, eps)])
            r = float(np.mean([episode(adapter, env, adaptive_lr=adaptive_lr) for _ in range(n_avg)]))
            eps_all.append(eps)
            rewards.append(r)
        rewards = np.array(rewards)
        normed = (rewards - rewards.mean()) / rewards.std() if rewards.std() > 1e-8 else rewards - rewards.mean()
        deltas = [np.zeros_like(p) for p in base]
        for r, eps in zip(normed, eps_all):
            for i in range(len(deltas)):
                deltas[i] += r * eps[i]
        base = tuple(b + (lr / (pop * sigma)) * d for b, d in zip(base, deltas))
        adapter.set(*base)
    return adapter


def run_seed(make_adapter, seed, gens=80, balls_per_episode=30, adaptive_lr=0.0):
    env = Catch(balls_per_episode=balls_per_episode, rng=np.random.default_rng(seed))
    adapter = make_adapter(seed)
    adapter = train_es(adapter, env, gens=gens, pop=24, n_avg=3, adaptive_lr=adaptive_lr, seed=seed)
    return evaluate(adapter, env, n=200, adaptive_lr=adaptive_lr)


def main():
    sample_path = 'transfer/sample_brains/v2_multiworld_overall_champion.json'
    trained = TBI.from_checkpoint(sample_path)
    has_ep = trained.episodic_slots is not None
    cap = trained.episodic_slots.shape[0] if has_ep else 0
    print(f'Loaded v2 brain: hidden={trained.hidden_size}, attn={trained.attention_weights is not None}, episodic={has_ep} (cap {cap})')

    SEEDS = list(range(20, 25))  # 5 seeds for the local validation
    GENS = 80
    ADAPTIVE_LR = 0.10
    BALLS = 30

    results = {}
    for mode_label, mode_lr in [('frozen', 0.0), ('adaptive', ADAPTIVE_LR)]:
        print(f'\n================  MODE: {mode_label.upper()}  ================')

        t0 = time.time()
        print('A) Direct linear policy (no brain) ...')
        results[f'direct_{mode_label}'] = [
            run_seed(lambda s: DirectPolicy(rng=np.random.default_rng(s)), s,
                     gens=GENS, balls_per_episode=BALLS, adaptive_lr=mode_lr)
            for s in SEEDS
        ]
        print(f'   {[round(x, 3) for x in results[f"direct_{mode_label}"]]}  [{time.time()-t0:.0f}s]')

        t0 = time.time()
        print('B) Random-init brain + adapter ...')
        def random_factory(s):
            rb = TBI.random(72, trained.hidden_size, 15, rng=np.random.default_rng(s + 100),
                            with_episodic=has_ep, capacity=cap)
            return CatchAdapter(rb, rng=np.random.default_rng(s + 1000))
        results[f'random_brain_{mode_label}'] = [
            run_seed(random_factory, s, gens=GENS, balls_per_episode=BALLS, adaptive_lr=mode_lr)
            for s in SEEDS
        ]
        print(f'   {[round(x, 3) for x in results[f"random_brain_{mode_label}"]]}  [{time.time()-t0:.0f}s]')

        t0 = time.time()
        print('C) Permuted v2 brain + adapter ...')
        def permuted_factory(s):
            pb = shuffle_brain(trained, rng=np.random.default_rng(s + 2000))
            return CatchAdapter(pb, rng=np.random.default_rng(s + 1000))
        results[f'permuted_{mode_label}'] = [
            run_seed(permuted_factory, s, gens=GENS, balls_per_episode=BALLS, adaptive_lr=mode_lr)
            for s in SEEDS
        ]
        print(f'   {[round(x, 3) for x in results[f"permuted_{mode_label}"]]}  [{time.time()-t0:.0f}s]')

        t0 = time.time()
        print('D) v2 multi-world brain + adapter ...')
        def trained_factory(s):
            # Fresh deep copy per seed in case adaptive learning mutates it.
            tb = TBI.from_checkpoint(sample_path)
            return CatchAdapter(tb, rng=np.random.default_rng(s + 1000))
        results[f'trained_{mode_label}'] = [
            run_seed(trained_factory, s, gens=GENS, balls_per_episode=BALLS, adaptive_lr=mode_lr)
            for s in SEEDS
        ]
        print(f'   {[round(x, 3) for x in results[f"trained_{mode_label}"]]}  [{time.time()-t0:.0f}s]')

    print('\n=== Final summary ===')
    print(f'  {"Condition":24s}  {"Frozen":>20s}  {"Adaptive":>20s}')
    for k, lbl in zip(['direct', 'random_brain', 'permuted', 'trained'],
                      ['Direct (no brain)', 'Random-init brain', 'Permuted trained brain', 'v2 trained brain']):
        f_arr = np.array(results[f'{k}_frozen'])
        a_arr = np.array(results[f'{k}_adaptive'])
        print(f'  {lbl:24s}  {f_arr.mean():+.3f} +/- {f_arr.std():.3f}      {a_arr.mean():+.3f} +/- {a_arr.std():.3f}')

    t_f = np.mean(results['trained_frozen'])
    t_a = np.mean(results['trained_adaptive'])
    print('\n  Key contrasts (frozen mode):')
    print(f'    trained - random_brain: {t_f - np.mean(results["random_brain_frozen"]):+.3f}')
    print(f'    trained - permuted:     {t_f - np.mean(results["permuted_frozen"]):+.3f}')
    print(f'    trained - direct:       {t_f - np.mean(results["direct_frozen"]):+.3f}')
    print('\n  Adaptive vs frozen (does brain plasticity help on Catch?):')
    print(f'    trained:      {t_a - t_f:+.3f}')
    print(f'    random_brain: {np.mean(results["random_brain_adaptive"]) - np.mean(results["random_brain_frozen"]):+.3f}')
    print(f'    permuted:     {np.mean(results["permuted_adaptive"]) - np.mean(results["permuted_frozen"]):+.3f}')


if __name__ == '__main__':
    main()
