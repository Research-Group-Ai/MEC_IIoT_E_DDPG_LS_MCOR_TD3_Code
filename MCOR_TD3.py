
from __future__ import annotations
import argparse
import copy
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np


@dataclass
class Config:
    devices: int = 16
    servers: int = 3
    cache_capacity: int = 8
    catalog_size: int = 32
    steps: int = 100
    seed: int = 42
    local_hz: float = 1e9
    server_hz: float = 20e9
    bandwidth_hz: float = 50e6
    transmit_dbm: float = 25.0
    channel_gain: float = 1e-8  # assumption
    noise_w: float = 1e-10  # assumption
    wired_bps: float = 1e9  # assumption
    forwarding_load: float = 5.0  # tasks/sec, assumption for Eq. 16
    retrieval_s: float = 0.001  # assumption
    gamma: float = 0.99
    learning_rate: float = 1e-5
    batch_size: int = 256
    replay_capacity: int = 120000
    policy_delay: int = 2
    tau: float = 0.005  # unspecified in Table 2
    target_noise: float = 0.2
    noise_clip: float = 0.5
    exploration_noise: float = 0.1
    warmup: int = 1000

    def validate(self):
        for key in ('devices', 'servers', 'catalog_size', 'steps', 'batch_size',
                    'replay_capacity', 'policy_delay'):
            if getattr(self, key) < 1:
                raise ValueError(f'{key} must be positive.')
        if not 0 <= self.cache_capacity <= self.catalog_size:
            raise ValueError('cache_capacity must be between 0 and catalog_size.')
        if self.batch_size > self.replay_capacity or self.warmup < 0:
            raise ValueError('Invalid replay/batch/warmup settings.')
        for key in ('local_hz', 'server_hz', 'bandwidth_hz', 'channel_gain',
                    'noise_w', 'wired_bps', 'learning_rate'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be finite and positive.')
        for key in ('forwarding_load', 'retrieval_s', 'target_noise',
                    'noise_clip', 'exploration_noise'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f'{key} must be finite and nonnegative.')
        if not math.isfinite(self.transmit_dbm):
            raise ValueError('transmit_dbm must be finite.')
        if not 0 < self.tau < 1 or not 0 <= self.gamma <= 1:
            raise ValueError('Require 0<tau<1 and 0<=gamma<=1.')
        if (9.6e6 / self.wired_bps) * self.forwarding_load >= 1:
            raise ValueError('Eq. 16 queue is unstable: expected_service*load >= 1.')


class MECEnvironment:
    """Explicit batch surrogate described in the module header; units are SI."""
    def __init__(self, cfg: Config):
        cfg.validate()
        self.cfg = cfg
        n, m, k = cfg.devices, cfg.servers, cfg.catalog_size
        self.association = np.arange(n) % m
        counts = np.bincount(self.association, minlength=m)
        power = 10 ** ((cfg.transmit_dbm - 30) / 10)
        self.rate = (cfg.bandwidth_hz * np.log2(1 + power*cfg.channel_gain/cfg.noise_w)
                     / counts[self.association])
        catalog_rng = np.random.default_rng(cfg.seed)
        self.sizes = catalog_rng.uniform(4e6, 9.6e6, k)
        self.cycles = catalog_rng.uniform(6e9, 14.4e9, k)
        self.priorities = catalog_rng.choice([1., 1.5, 1.8, 2., 2.6], k)
        self.action_dim = n * (m + 2)
        self.state_dim = n * (3+k) + cfg.cache_capacity * (k+1)
        self.reset(cfg.seed)

    def reset(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.cache = []  # least recent -> most recent; ordered state
        self.t = 0
        self.ids = self.rng.integers(self.cfg.catalog_size, size=self.cfg.devices)
        return self.state()

    def state(self):
        c, ids = self.cfg, self.ids
        features = np.column_stack((self.sizes[ids]/9.6e6, self.cycles[ids]/14.4e9,
                                    self.priorities[ids]/2.6))
        tasks = np.concatenate((features, np.eye(c.catalog_size)[ids]), axis=1)
        slots = np.zeros((c.cache_capacity, c.catalog_size+1))
        slots[:, -1] = 1  # empty cache slot
        for i, key in enumerate(self.cache):
            slots[i] = 0
            slots[i, key] = 1
        return np.concatenate((tasks.ravel(), slots.ravel())).astype(np.float32)

    def decode(self, action):
        c = self.cfg
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise ValueError('Action has invalid shape or non-finite coordinates.')
        if (np.abs(action) > 1+1e-6).any():
            raise ValueError('Encoded actions must lie in [-1,1].')
        rows = np.clip(action, -1, 1).reshape(c.devices, c.servers+2)
        destination = rows[:, :c.servers+1].argmax(axis=1)
        hits = np.isin(self.ids, self.cache) & (destination > 0)
        frequency = np.zeros(c.devices)
        weights = (rows[:, -1]+1)/2 + 1e-3
        for m in range(c.servers):
            active = (destination == m+1) & ~hits
            if active.any():
                frequency[active] = c.server_hz * weights[active]/weights[active].sum()
        return destination, frequency, hits

    def step(self, action):
        if self.t >= self.cfg.steps:
            raise RuntimeError('Episode finished; call reset before step.')
        c = self.cfg
        destination, frequency, hits = self.decode(action)
        size, cycles = self.sizes[self.ids], self.cycles[self.ids]
        delay = cycles/c.local_hz
        uplink = size/self.rate
        for n in range(c.devices):
            if destination[n] == 0:
                continue
            if hits[n]:
                delay[n] = uplink[n] + c.retrieval_s
                continue
            delay[n] = uplink[n] + cycles[n]/frequency[n]
            if destination[n]-1 != self.association[n]:
                service = size[n]/c.wired_bps
                queue_term = service/(1-service*c.forwarding_load)  # literal Eq.16
                delay[n] += queue_term + service  # Eq.18
        # Deterministic device-order LRU updates only after batch completion.
        if c.cache_capacity:
            for n in range(c.devices):
                if destination[n] > 0:
                    key = int(self.ids[n])
                    if key in self.cache:
                        self.cache.remove(key)
                    self.cache.append(key)
                    self.cache = self.cache[-c.cache_capacity:]
        reward = -float(delay.max())
        info = {'mean_delay_s': float(delay.mean()), 'max_delay_s': float(delay.max()),
                'cache_hits': int(hits.sum()), 'requests': c.devices,
                'offloaded': int((destination > 0).sum())}
        self.t += 1
        self.ids = self.rng.integers(c.catalog_size, size=c.devices)
        # Finite-horizon terminal transition: no bootstrap at episode end.
        # Remaining horizon is included in the agent observation below.
        return self.state(), reward, self.t >= c.steps, info

    def observation(self, state):
        return np.append(state, 1-self.t/self.cfg.steps).astype(np.float32)


def baseline_action(env, mode, rng):
    c = env.cfg
    rows = np.full((c.devices, c.servers+2), -1., dtype=np.float32)
    rows[:, -1] = 0  # equal allocation weights
    if mode == 'random':
        destination = rng.integers(c.servers+1, size=c.devices)
    elif mode == 'local':
        destination = np.zeros(c.devices, dtype=int)
    elif mode == 'greedy':
        destination = np.zeros(c.devices, dtype=int)
        assumed_hz = c.server_hz/max(1, c.devices/c.servers)
        for n, key in enumerate(env.ids):
            candidates = [env.cycles[key]/c.local_hz]
            for m in range(c.servers):
                d = env.sizes[key]/env.rate[n]
                if key in env.cache:
                    d += c.retrieval_s
                else:
                    d += env.cycles[key]/assumed_hz
                    if m != env.association[n]:
                        service = env.sizes[key]/c.wired_bps
                        d += service + service/(1-service*c.forwarding_load)
                candidates.append(d)
            destination[n] = np.argmin(candidates)
    else:
        raise ValueError(mode)
    rows[np.arange(c.devices), destination] = 1
    return rows.ravel()


class ReplayBuffer:
    def __init__(self, cfg, state_dim, action_dim):
        self.capacity, self.position, self.size = cfg.replay_capacity, 0, 0
        self.rng = np.random.default_rng(cfg.seed+11)
        self.s = np.empty((self.capacity, state_dim), np.float32)
        self.a = np.empty((self.capacity, action_dim), np.float32)
        self.ns = np.empty_like(self.s)
        self.r = np.empty((self.capacity, 1), np.float32)
        self.done = np.empty_like(self.r)

    def add(self, s, a, r, ns, done):
        i = self.position
        self.s[i], self.a[i], self.r[i], self.ns[i], self.done[i] = s, a, r, ns, done
        self.position = (i+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample(self, count):
        if self.size < count:
            raise ValueError('Not enough replay samples.')
        i = self.rng.choice(self.size, count, replace=False)
        return [x[i] for x in (self.s, self.a, self.r, self.ns, self.done)]


class TD3Agent:
    def __init__(self, cfg, state_dim, action_dim, device='cpu'):
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError('Training needs PyTorch: pip install torch') from exc
        self.torch, self.cfg, self.device = torch, cfg, device
        torch.manual_seed(cfg.seed)
        self.updates = 0
        def mlp(widths, tanh=False):
            layers = []
            for i, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
                layers.append(nn.Linear(left, right))
                if i < len(widths)-2:
                    layers.append(nn.ReLU())
            if tanh:
                layers.append(nn.Tanh())
            return nn.Sequential(*layers).to(device)
        self.actor = mlp([state_dim, 256, 128, action_dim], True)
        self.q1 = mlp([state_dim+action_dim, 256, 256, 128, 1])
        self.q2 = mlp([state_dim+action_dim, 256, 256, 128, 1])
        self.targets = [copy.deepcopy(x).requires_grad_(False)
                        for x in (self.actor, self.q1, self.q2)]
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.learning_rate)
        self.critic_opt = torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()),
                                          lr=cfg.learning_rate)

    def action(self, state):
        with self.torch.no_grad():
            return self.actor(self.torch.as_tensor(state, device=self.device)[None])[0].cpu().numpy()

    def update(self, replay):
        t, c = self.torch, self.cfg
        s, a, r, ns, done = [t.as_tensor(x, device=self.device)
                             for x in replay.sample(c.batch_size)]
        actor_t, q1_t, q2_t = self.targets
        with t.no_grad():
            noise = (t.randn_like(a)*c.target_noise).clamp(-c.noise_clip, c.noise_clip)
            na = (actor_t(ns)+noise).clamp(-1, 1)
            nxt = t.cat([ns, na], 1)
            y = r+c.gamma*(1-done)*t.minimum(q1_t(nxt), q2_t(nxt))
        sa = t.cat([s, a], 1)
        loss = ((self.q1(sa)-y)**2).mean()+((self.q2(sa)-y)**2).mean()
        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_opt.step()
        self.updates += 1
        actor_value = None
        if self.updates % c.policy_delay == 0:
            self.q1.requires_grad_(False)
            try:
                actor_loss = -self.q1(t.cat([s, self.actor(s)], 1)).mean()
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                self.actor_opt.step()
                actor_value = float(actor_loss.item())
            finally:
                self.q1.requires_grad_(True)
            with t.no_grad():
                for target, online in zip(self.targets, (self.actor, self.q1, self.q2)):
                    for dst, src in zip(target.parameters(), online.parameters()):
                        dst.lerp_(src, c.tau)
        return float(loss.item()), actor_value

    def save(self, path):
        self.torch.save({'config': asdict(self.cfg), 'actor': self.actor.state_dict()}, path)


def write_csv(path, rows):
    if rows:
        with Path(path).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def train(cfg, episodes, output, device):
    env = MECEnvironment(cfg)
    agent = TD3Agent(cfg, env.state_dim+1, env.action_dim, device)
    replay = ReplayBuffer(cfg, env.state_dim+1, env.action_dim)
    rng = np.random.default_rng(cfg.seed+13)
    global_step, history = 0, []
    for episode in range(episodes):
        state = env.observation(env.reset(cfg.seed+episode))
        total_reward, delays, losses, hits, offloaded = 0., [], [], 0, 0
        for _ in range(cfg.steps):
            if global_step < cfg.warmup:
                action = rng.uniform(-1, 1, env.action_dim).astype(np.float32)
            else:
                action = np.clip(agent.action(state)+rng.normal(
                    0, cfg.exploration_noise, env.action_dim), -1, 1).astype(np.float32)
            ns, reward, done, info = env.step(action)
            ns = env.observation(ns)
            replay.add(state, action, reward, ns, done)
            global_step += 1
            if global_step >= cfg.warmup and replay.size >= cfg.batch_size:
                loss, _ = agent.update(replay)
                losses.append(loss)
            total_reward += reward
            delays.append(info['mean_delay_s'])
            hits += info['cache_hits']
            offloaded += info['offloaded']
            state = ns
        row = {'episode': episode+1, 'return': total_reward,
               'mean_delay_s': float(np.mean(delays)),
               'mean_slot_max_delay_s': -total_reward/cfg.steps,
               'cache_hit_ratio_all_requests': hits/(cfg.steps*cfg.devices),
               'offload_ratio': offloaded/(cfg.steps*cfg.devices),
               'critic_loss': float(np.mean(losses)) if losses else '',
               'critic_updates': agent.updates}
        history.append(row)
        print(f'Episode {episode+1}/{episodes}: return={total_reward:.3f}, '
              f'mean delay={row["mean_delay_s"]:.4f}s, updates={agent.updates}', flush=True)
        write_csv(output/'training.csv', history)
        agent.save(output/'agent.pt')
    if agent.updates == 0:
        print('No learning updates occurred; increase episodes/steps or reduce warmup/batch.')
    return agent


def evaluate(cfg, episodes, agent=None):
    modes = ['local', 'random', 'greedy'] + (['td3'] if agent is not None else [])
    rows = []
    # Same task catalog, request streams and episode seeds for every policy.
    # Caches diverge only as a consequence of each policy's own actions.
    for mode in modes:
        env = MECEnvironment(cfg)
        for episode in range(episodes):
            seed = cfg.seed+100000+episode  # held out from training
            state = env.observation(env.reset(seed))
            rng = np.random.default_rng(seed+200000)
            reward_sum, mean_delays, hits, offloaded = 0., [], 0, 0
            for _ in range(cfg.steps):
                action = agent.action(state) if mode == 'td3' else baseline_action(env, mode, rng)
                ns, reward, done, info = env.step(action)
                state = env.observation(ns)
                reward_sum += reward
                mean_delays.append(info['mean_delay_s'])
                hits += info['cache_hits']
                offloaded += info['offloaded']
            rows.append({'policy': mode, 'episode': episode+1, 'seed': seed,
                         'return': reward_sum, 'mean_delay_s': float(np.mean(mean_delays)),
                         'mean_slot_max_delay_s': -reward_sum/cfg.steps,
                         'cache_hit_ratio_all_requests': hits/(cfg.steps*cfg.devices),
                         'cache_hit_ratio_offloaded_requests': hits/offloaded if offloaded else 0.,
                         'offload_ratio': offloaded/(cfg.steps*cfg.devices)})
        print(mode, 'mean delay (s):', np.mean([r['mean_delay_s'] for r in rows if r['policy']==mode]))
    return rows


def self_test():
    c = Config(devices=6, servers=3, steps=4, batch_size=2, replay_capacity=10)
    env = MECEnvironment(c)
    s = env.reset(1)
    assert s.shape == (env.state_dim,)
    rng = np.random.default_rng(1)
    # Allocation constraint on all servers; also cover cache hits and local mode.
    for _ in range(c.steps):
        action = rng.uniform(-1, 1, env.action_dim)
        dst, freq, hits = env.decode(action)
        for m in range(c.servers):
            assert freq[dst == m+1].sum() <= c.server_hz*(1+1e-12)
        assert (freq >= 0).all() and (freq[hits] == 0).all()
        ns, r, done, info = env.step(action)
        assert np.isfinite(ns).all() and r == -info['max_delay_s']
    assert done
    env.reset(4)
    expected = env.cycles[env.ids]/c.local_hz
    _, r, _, info = env.step(baseline_action(env, 'local', rng))
    np.testing.assert_allclose(info['mean_delay_s'], expected.mean())
    np.testing.assert_allclose(r, -expected.max())
    # Cache eligibility and full cache-hit batch.
    env.reset(1)
    env.ids[:] = 0
    env.cache = [0]
    action = baseline_action(env, 'local', rng).reshape(c.devices, c.servers+2)
    action[:, 0], action[:, 1] = -1, 1
    _, freq, hits = env.decode(action.ravel())
    assert hits.all() and (freq == 0).all()
    _, _, _, info = env.step(action.ravel())
    np.testing.assert_allclose(info['mean_delay_s'], (env.sizes[0]/env.rate+c.retrieval_s).mean())
    # Replay and repeatable held-out evaluation.
    replay = ReplayBuffer(c, 2, 3)
    for i in range(12):
        replay.add([i, i], [0, 0, 0], -i, [i+1, i+1], False)
    assert replay.size == 10 and replay.sample(2)[0].shape == (2, 2)
    assert evaluate(c, 1) == evaluate(c, 1)
    print('Environment, allocation, reward, cache, replay and reproducibility checks passed.')
    try:
        import torch
    except ImportError:
        print('PyTorch unavailable: neural-network/update checks SKIPPED.')
        return
    torch.set_num_threads(1)
    agent = TD3Agent(c, 2, 3)
    original = [p.detach().clone() for p in agent.actor.parameters()]
    l1, a1 = agent.update(replay)
    assert a1 is None and all(torch.equal(a, b) for a, b in zip(original, agent.actor.parameters()))
    l2, a2 = agent.update(replay)
    assert all(math.isfinite(x) for x in [l1, l2, a2])
    assert any(not torch.equal(a, b) for a, b in zip(original, agent.actor.parameters()))
    assert all(p.grad is None for target in agent.targets for p in target.parameters())
    print('Neural-network, delayed update and target-gradient checks passed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=['train', 'evaluate', 'self-test'], default='train')
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--eval-episodes', type=int, default=5)
    parser.add_argument('--output', default='results_mcor')
    parser.add_argument('--checkpoint')
    parser.add_argument('--device', default='cpu')
    # Expose every simulator/learning assumption and record actual values.
    defaults = Config()
    for name, value in asdict(defaults).items():
        parser.add_argument('--'+name.replace('_', '-'), type=type(value), default=value)
    args = parser.parse_args()
    if args.mode == 'self-test':
        self_test()
        return
    if args.episodes < 1 or args.eval_episodes < 1:
        parser.error('Episode counts must be positive.')
    cfg = Config(**{name: getattr(args, name) for name in asdict(defaults)})
    agent = None
    if args.checkpoint:
        if args.mode != 'evaluate':
            parser.error('--checkpoint is for evaluation only, not training resumption.')
        import torch
        saved = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        cfg = Config(**saved['config'])  # exact environment/encoding used to train
        env = MECEnvironment(cfg)
        agent = TD3Agent(cfg, env.state_dim+1, env.action_dim, args.device)
        agent.actor.load_state_dict(saved['actor'])
        print('Using saved environment configuration; other Config flags are ignored.')
    cfg.validate()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'config.json').write_text(json.dumps(asdict(cfg), indent=2))
    (output/'IMPLEMENTATION_NOTES.txt').write_text(__doc__)
    if args.mode == 'train':
        agent = train(cfg, args.episodes, output, args.device)
        rows = evaluate(cfg, args.eval_episodes, agent)
    else:
        rows = evaluate(cfg, args.episodes, agent)
    write_csv(output/'evaluation.csv', rows)
    print('Saved outputs to', output.resolve())


if __name__ == '__main__':
    main()
