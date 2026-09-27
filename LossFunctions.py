
from __future__ import annotations
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _column(x: Tensor, name: str) -> Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(1)
    if x.ndim != 2 or x.shape[1] != 1 or x.shape[0] == 0:
        raise ValueError(f'{name} must have nonempty shape [B] or [B, 1].')
    if not torch.isfinite(x).all():
        raise ValueError(f'{name} contains non-finite values.')
    return x


def _match(*xs: Tensor) -> None:
    if any(x.shape != xs[0].shape or x.device != xs[0].device for x in xs):
        raise ValueError('Batch shapes and devices must agree; no broadcasting.')


class MECMaxDelayReward(nn.Module):
    """Eq. 30: -max delay over devices. Input [B,N], output [B,1].

    Supply nonnegative total completion delays from Eq. 19 in consistent units.
    Cache hits should include actual retrieval delay, not assumed zero delay.
    """
    def forward(self, delays: Tensor) -> Tensor:
        if delays.ndim != 2 or 0 in delays.shape or not delays.is_floating_point():
            raise ValueError('delays must be a nonempty floating-point [B,N] tensor.')
        if not torch.isfinite(delays).all() or (delays < 0).any():
            raise ValueError('Delays must be finite and nonnegative.')
        return -delays.amax(dim=1, keepdim=True)


class DDPGCriticLoss(nn.Module):
    """Generic DDPG TD loss; supply the E-DDPG-LS reward externally."""
    def __init__(self, gamma: float = 0.99):
        super().__init__()
        if not 0 <= gamma <= 1:
            raise ValueError('gamma must be in [0,1].')
        self.gamma = gamma

    @torch.no_grad()
    def target(self, reward: Tensor, terminated: Tensor, next_q: Tensor) -> Tensor:
        reward = _column(reward, 'reward')
        terminated = _column(terminated, 'terminated')
        next_q = _column(next_q, 'next_q')
        _match(reward, terminated, next_q)
        if not ((terminated == 0) | (terminated == 1)).all():
            raise ValueError('terminated must contain booleans or 0/1.')
        return reward + self.gamma * torch.where(
            terminated.bool(), torch.zeros_like(next_q), next_q)

    def forward(self, current_q: Tensor, target_q: Tensor) -> Tensor:
        current_q, target_q = _column(current_q, 'current_q'), _column(target_q, 'target_q')
        _match(current_q, target_q)
        return F.mse_loss(current_q, target_q.detach())


class TD3CriticLoss(DDPGCriticLoss):
    """Eqs. 32-33: minimum target Q and sum of the two critic MSEs."""
    @torch.no_grad()
    def target(self, reward: Tensor, terminated: Tensor,
               next_q1: Tensor, next_q2: Tensor) -> Tensor:
        next_q1, next_q2 = _column(next_q1, 'next_q1'), _column(next_q2, 'next_q2')
        _match(next_q1, next_q2)
        return super().target(reward, terminated, torch.minimum(next_q1, next_q2))

    def forward(self, q1: Tensor, q2: Tensor, target_q: Tensor) -> tuple[Tensor, dict]:
        loss1 = super().forward(q1, target_q)
        loss2 = super().forward(q2, target_q)
        total = loss1 + loss2
        return total, {'critic_loss': total.detach(), 'critic1_loss': loss1.detach(),
                       'critic2_loss': loss2.detach()}


class DeterministicActorLoss(nn.Module):
    """Eq. 36: minimize -mean(Q1(s,actor(s))). Also usable for DDPG.

    Do not detach policy actions/Q values. Freeze critic parameters during the
    actor update but do not put the critic forward pass inside no_grad().
    """
    def forward(self, policy_q: Tensor) -> Tensor:
        return -_column(policy_q, 'policy_q').mean()


@torch.no_grad()
def smooth_target_action(action: Tensor, *, noise_std: float, noise_clip: float,
                         action_low: float, action_high: float) -> Tensor:
    """Eq. 37; scalar bounds apply to all encoded action coordinates.

    For normalized Tanh output use [-1,1], not physical GHz allocations.
    """
    if not all(math.isfinite(v) for v in (noise_std, noise_clip, action_low, action_high)):
        raise ValueError('Settings must be finite.')
    if noise_std < 0 or noise_clip < 0 or action_low >= action_high:
        raise ValueError('Require nonnegative noise settings and low < high.')
    if action.ndim != 2 or 0 in action.shape or not action.is_floating_point():
        raise ValueError('action must be a nonempty floating-point [B,A] tensor.')
    if not torch.isfinite(action).all():
        raise ValueError('action contains non-finite values.')
    noise = (torch.randn_like(action) * noise_std).clamp(-noise_clip, noise_clip)
    return (action + noise).clamp(action_low, action_high)


@torch.no_grad()
def soft_update(target: nn.Module, online: nn.Module, *, tau: float) -> None:
    """Eqs. 34-35: target <- tau*online + (1-tau)*target.

    For the paper's buffer-free MLPs. Initialize targets using load_state_dict
    and disable target parameter gradients before training. Call for actor and
    both critics after each delayed actor update.
    """
    if not 0 < tau < 1:
        raise ValueError('Supply the actual tau in (0,1).')
    if target is online or list(target.buffers()) or list(online.buffers()):
        raise ValueError('Expected distinct, buffer-free target and online MLPs.')
    dst, src = dict(target.named_parameters()), dict(online.named_parameters())
    if not dst or dst.keys() != src.keys():
        raise ValueError('Networks must have matching named parameters.')
    for name, p in dst.items():
        q = src[name]
        if p.shape != q.shape or p.device != q.device or p.dtype != q.dtype:
            raise ValueError(f'Parameter mismatch: {name}')
        if p.data_ptr() == q.data_ptr():
            raise ValueError('Target parameters must not alias online parameters.')
    for name, p in dst.items():
        p.lerp_(src[name], tau)


def should_update_actor(critic_updates: int, policy_delay: int = 2) -> bool:
    """Global count of completed critic updates; Table 2 specifies u=2."""
    if not isinstance(policy_delay, int) or policy_delay < 1:
        raise ValueError('policy_delay must be a positive integer.')
    if not isinstance(critic_updates, int) or critic_updates < 0:
        raise ValueError('critic_updates must be a nonnegative integer.')
    return critic_updates > 0 and critic_updates % policy_delay == 0


def _self_test() -> None:
    torch.manual_seed(7)
    reward = MECMaxDelayReward()(torch.tensor([[0.2, 0.5], [0.3, 0.1]]))
    torch.testing.assert_close(reward, torch.tensor([[-0.5], [-0.3]]))
    objective = TD3CriticLoss()
    nq1 = torch.tensor([[2.0], [9.0]], requires_grad=True)
    nq2 = torch.tensor([[3.0], [8.0]], requires_grad=True)
    y = objective.target(reward, torch.tensor([False, True]), nq1, nq2)
    torch.testing.assert_close(y, torch.tensor([[1.48], [-0.3]]))
    assert not y.requires_grad
    q1, q2 = torch.zeros(2, 1, requires_grad=True), torch.ones(2, 1, requires_grad=True)
    loss, metrics = objective(q1, q2, y)
    torch.testing.assert_close(loss, ((q1-y)**2).mean()+((q2-y)**2).mean())
    loss.backward()
    assert q1.grad is not None and q2.grad is not None and nq1.grad is None
    actor, critic = nn.Linear(3, 2), nn.Linear(5, 1)
    critic.requires_grad_(False)
    states = torch.randn(4, 3)
    DeterministicActorLoss()(critic(torch.cat([states, actor(states)], dim=1))).backward()
    assert actor.weight.grad is not None and actor.weight.grad.abs().sum() > 0
    assert critic.weight.grad is None
    # Illustrative checks only: these noise/tau values are NOT paper settings.
    a = smooth_target_action(torch.zeros(4, 2), noise_std=0.2, noise_clip=0.1,
                             action_low=-1, action_high=1)
    assert (a.abs() <= 0.1).all() and not a.requires_grad
    online, target = nn.Linear(2, 1), nn.Linear(2, 1)
    with torch.no_grad():
        for p in online.parameters():
            p.fill_(1)
        for p in target.parameters():
            p.zero_()
    soft_update(target, online, tau=0.1)
    for p in target.parameters():
        torch.testing.assert_close(p, torch.full_like(p, 0.1))
    assert not should_update_actor(1) and should_update_actor(2)
    try:
        objective(q1, q2, torch.zeros(3, 1))
    except ValueError:
        pass
    else:
        raise AssertionError('Mismatched batches must not broadcast.')
    print('All numerical and gradient checks passed.')
    print({k: v.item() for k, v in metrics.items()})


if __name__ == '__main__':
    _self_test()
