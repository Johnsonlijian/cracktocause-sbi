"""Amortized neural posterior estimation (sbi) for crack-line maps.

Parameters live in a box [low, high]; the flow is trained on z = logit((theta - low) / (high - low)),
whose prior is a product of standard logistic distributions. The posterior support then matches the
prior box exactly, so sampling needs no rejection step.
"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, TransformedDistribution, Uniform
from torch.distributions.transforms import SigmoidTransform


class Box:
    def __init__(self, low, high, names):
        self.low = np.asarray(low, float)
        self.high = np.asarray(high, float)
        self.names = list(names)

    def to_z(self, theta, eps=1e-6):
        u = np.clip((np.asarray(theta, float) - self.low) / (self.high - self.low), eps, 1 - eps)
        return np.log(u) - np.log1p(-u)

    def to_theta(self, z):
        return self.low + (self.high - self.low) / (1.0 + np.exp(-np.asarray(z, float)))

    def z_prior(self, device='cpu'):
        d = len(self.low)
        base = Uniform(torch.zeros(d, device=device), torch.ones(d, device=device))
        return Independent(TransformedDistribution(base, [SigmoidTransform().inv]), 1)


class CNN(nn.Module):
    """Summary network for binary crack-line maps of shape (ny, nx)."""

    def __init__(self, shape, out=96, width=(16, 32, 64, 96)):
        super().__init__()
        self.shape = tuple(shape)
        c = (1,) + tuple(width)
        layers = []
        for a, b in zip(c[:-1], c[1:]):
            layers += [nn.Conv2d(a, b, 3, padding=1), nn.GELU(), nn.MaxPool2d(2)]
        self.conv = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((6, 6)))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c[-1] * 36, 256), nn.GELU(), nn.Linear(256, out))

    def forward(self, x):
        return self.head(self.conv(x.reshape(-1, 1, *self.shape)))


class FeatureNet(nn.Module):
    """Summary network on fixed crack-map features: crack density per column band and per row band
    (pooled over `bx` x `by` pixels), plus the total crack length, log-scaled, followed by an MLP.
    More data-efficient than a CNN when only a few hundred simulations are available.
    With n_extra > 0 the last n_extra entries of each input row are scalar observations appended to
    the features (here: log of the widest crack opening and an indicator that it was measured)."""

    def __init__(self, shape, bx=5, by=10, out=64, hidden=128, dropout=0.1, cy=None, n_extra=0):
        super().__init__()
        self.shape, self.bx, self.by, self.n_extra = tuple(shape), bx, by, n_extra
        self.cy = cy or 4 * by                            # rows per cell of the coarse grid
        ny, nx = shape
        self.npix = ny * nx
        nf = int(np.ceil(nx / bx)) + int(np.ceil(ny / by)) + 1 + int(np.ceil(nx / bx)) * int(np.ceil(ny / self.cy))
        self.mlp = nn.Sequential(nn.Linear(nf + n_extra, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out))

    def features(self, x):
        img = x[:, :self.npix].reshape(-1, 1, *self.shape)
        cols = nn.functional.avg_pool2d(img.sum(2, keepdim=True), (1, self.bx), ceil_mode=True).flatten(1)
        rows = nn.functional.avg_pool2d(img.sum(3, keepdim=True), (self.by, 1), ceil_mode=True).flatten(1)
        coarse = nn.functional.avg_pool2d(img, (self.cy, self.bx), ceil_mode=True).flatten(1) * (self.cy * self.bx)
        tot = img.flatten(1).sum(1, keepdim=True)
        f = torch.log1p(torch.cat([cols * self.bx, rows * self.by, tot, coarse], 1))
        return torch.cat([f, x[:, self.npix:]], 1) if self.n_extra else f

    def forward(self, x):
        return self.mlp(self.features(x.reshape(len(x), -1)))


def augment(x, shape, gen, shift=2, drop=0.25, spur=0.003, p_dilate=0.3):
    """Inspection-like corruption of binary maps x (B, ny*nx): rigid shift of up to `shift` pixels,
    pixel dropout (missed crack segments), spurious pixels (false detections), occasional dilation
    (thicker traced lines)."""
    B = x.shape[0]
    img = x.reshape(B, 1, *shape)
    dx, dy = int(gen.integers(-shift, shift + 1)), int(gen.integers(-shift, shift + 1))
    img = torch.roll(img, shifts=(dy, dx), dims=(2, 3))
    img = img * (torch.rand_like(img) > float(gen.uniform(0.0, drop))).float()
    img = torch.clamp(img + (torch.rand_like(img) < float(gen.uniform(0.0, spur))).float(), 0, 1)
    if gen.random() < p_dilate:
        img = nn.functional.max_pool2d(img, 3, stride=1, padding=1)
    return img.reshape(B, -1)


def train(theta, maps, box, shape, epochs=300, n_aug=4, seed=0, device=None, hidden=96, transforms=6,
          batch=256, lr=5e-4, show=True):
    """theta (N, D) in the box, maps (N, ny, nx) uint8. Returns (posterior in z-space, estimator)."""
    from sbi.inference import NPE
    from sbi.neural_nets import posterior_nn
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    gen = np.random.default_rng(seed)
    torch.manual_seed(seed)
    x = torch.tensor(maps.reshape(len(maps), -1), dtype=torch.float32)
    z = torch.tensor(box.to_z(theta), dtype=torch.float32)
    xs = [x] + [augment(x, shape, gen) for _ in range(n_aug)]
    x_tr = torch.cat(xs)
    z_tr = z.repeat(len(xs), 1)
    builder = posterior_nn(model='nsf', embedding_net=CNN(shape), hidden_features=hidden,
                           num_transforms=transforms, z_score_x='none', z_score_theta='independent')
    inf = NPE(prior=box.z_prior(device), density_estimator=builder, device=device, show_progress_bars=show)
    inf.append_simulations(z_tr.to(device), x_tr.to(device))
    est = inf.train(max_num_epochs=epochs, training_batch_size=batch, learning_rate=lr,
                    show_train_summary=show)
    post = inf.build_posterior(est)
    return post, est, inf


def augment_batch(img, gen, shift=2, drop=0.25, spur=0.003, p_dilate=0.3, block=1):
    """Inspection-noise corruption of a float batch (B, 1, ny, nx) on the device (see `augment`).
    With block > 1, dropout removes blocks of block x block pixels (missed crack segments) instead of
    single pixels."""
    dx, dy = int(gen.integers(-shift, shift + 1)), int(gen.integers(-shift, shift + 1))
    img = torch.roll(img, shifts=(dy, dx), dims=(2, 3))
    p = float(gen.uniform(0.0, drop))
    if block > 1:
        B, _, ny, nx = img.shape
        keep = torch.rand(B, 1, -(-ny // block), -(-nx // block), device=img.device) > p
        keep = keep.repeat_interleave(block, 2).repeat_interleave(block, 3)[:, :, :ny, :nx]
        img = img * keep.float()
    else:
        img = img * (torch.rand_like(img) > p).float()
    if spur > 0:
        img = torch.clamp(img + (torch.rand_like(img) < float(gen.uniform(0.0, spur))).float(), 0, 1)
    if gen.random() < p_dilate:
        img = nn.functional.max_pool2d(img, 3, stride=1, padding=1)
    return img


def train_streaming(theta, maps, groups, box, shape, epochs=200, patience=15, val_frac=0.1, seed=0,
                    device=None, hidden=96, transforms=6, batch=256, lr=5e-4, p_clean=0.2, log=print,
                    embedding='features', weight_decay=1e-4, z_noise=0.0, emb_kw=None, aug_kw=None,
                    model='nsf', extra=None, p_mask=0.3, extra_noise=0.2):
    """Memory-lean NPE training: maps stay uint8 on the device and every batch is augmented on the
    fly (a fraction p_clean of the batches is left clean); validation split by simulation (groups);
    early stopping on the validation loss. Returns (posterior in z-space, estimator, history).

    z_noise: standard deviation of Gaussian noise added to the training parameters (in units of
    their standard deviation in z-space); a scalar, or one value per parameter. All maps of a simulation share its non-staged parameters,
    so the training set contains only as many distinct parameter vectors as simulations, and a
    flexible flow learns density spikes on them; the noise smooths the target (noise
    regularization of conditional density estimators, Rothfuss et al. 2019) and broadens the
    posterior by at most z_noise standard deviations.

    extra: optional (N, 2) scalar observations per map, [log(widest crack opening + 0.05 mm), 1] when
    measured and [0, 0] when not (simulations without widths). During training a measured value is
    hidden with probability p_mask and perturbed by N(0, extra_noise^2) (relative measurement error),
    so that one estimator serves maps with and without a width measurement."""
    from sbi.inference.posteriors import DirectPosterior
    from sbi.neural_nets import posterior_nn
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    gen = np.random.default_rng(seed)
    torch.manual_seed(seed)
    sims = np.unique(groups)
    val_sims = gen.choice(sims, size=max(1, int(val_frac * len(sims))), replace=False)
    val = np.isin(groups, val_sims)
    X = torch.tensor(np.asarray(maps, np.uint8), device=device)              # (N, ny, nx) uint8
    Z = torch.tensor(box.to_z(theta), dtype=torch.float32, device=device)
    i_tr, i_va = np.flatnonzero(~val), np.flatnonzero(val)
    E = None if extra is None else torch.tensor(np.asarray(extra, np.float32), device=device)

    def xb(idx, aug, train=False):
        img = X[idx].float().unsqueeze(1)
        if aug:
            img = augment_batch(img, gen, **(aug_kw or {}))
        img = img.reshape(len(idx), -1)
        if E is None:
            return img
        e = E[idx].clone()
        if train:
            measured = e[:, 1] > 0.5
            e[:, 0] = torch.where(measured, e[:, 0] + extra_noise * torch.randn_like(e[:, 0]), e[:, 0])
            hide = measured & (torch.rand_like(e[:, 0]) < p_mask)
            e[hide] = 0.0
        return torch.cat([img, e], 1)

    emb_kw = dict(emb_kw or {})
    if E is not None:
        emb_kw['n_extra'] = E.shape[1]
    net = FeatureNet(shape, **emb_kw) if embedding == 'features' else CNN(shape, **emb_kw)
    nn_kw = dict(hidden_features=hidden) if model == 'mdn' else dict(hidden_features=hidden, num_transforms=transforms)
    builder = posterior_nn(model=model, embedding_net=net, z_score_x='none', z_score_theta='independent', **nn_kw)
    probe = gen.choice(i_tr, size=min(2048, len(i_tr)), replace=False)
    est = builder(Z[probe].cpu(), xb(probe, False).cpu()).to(device)     # (mdn probes the net on cpu)
    opt = torch.optim.AdamW(est.parameters(), lr=lr, weight_decay=weight_decay)
    z_sd = Z[i_tr].std(0)
    # noise level per parameter: a scalar applies to all parameters, a sequence gives one level each
    zn = torch.as_tensor(np.broadcast_to(np.asarray(z_noise, np.float32), (Z.shape[1],)).copy(), device=device)
    noisy = bool((zn > 0).any())
    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(epochs):
        est.train()
        perm = gen.permutation(i_tr)
        tot = 0.0
        for k in range(0, len(perm), batch):
            idx = perm[k:k + batch]
            zb = Z[idx] + zn * z_sd * torch.randn_like(Z[idx]) if noisy else Z[idx]
            loss = est.loss(zb, condition=xb(idx, gen.random() > p_clean, train=True)).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(est.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()) * len(idx)
        est.eval()
        with torch.no_grad():
            vl = sum(float(est.loss(Z[i_va[k:k + 1024]], condition=xb(i_va[k:k + 1024], False)).sum())
                     for k in range(0, len(i_va), 1024)) / max(len(i_va), 1)
        hist.append((tot / len(i_tr), vl))
        if vl < best - 1e-3:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in est.state_dict().items()}
        else:
            bad += 1
        log(f'epoch {ep + 1}: train {tot / len(i_tr):.3f} val {vl:.3f}' + (' *' if bad == 0 else ''))
        if bad >= patience:
            break
    est.load_state_dict(best_state)
    est.eval()
    post = DirectPosterior(posterior_estimator=est, prior=box.z_prior(device))
    return post, est, hist


class EnsemblePosterior:
    """Equal-weight mixture of independently trained posterior estimators (a deep ensemble), which
    reduces the variance of the estimate and makes it more conservative (Hermans et al. 2022).
    Implements the sample_batched interface used by `sample`."""

    def __init__(self, posts):
        self.posts = list(posts)

    def sample_batched(self, shape, x, show_progress_bars=False):
        n, K = shape[0], len(self.posts)
        ns = [n // K + (1 if k < n % K else 0) for k in range(K)]
        return torch.cat([p.sample_batched((m,), x=x, show_progress_bars=show_progress_bars)
                          for p, m in zip(self.posts, ns) if m > 0], 0)


def as_input(maps, extra=None):
    """Network input rows: flattened maps, followed by the scalar observations if given."""
    x = np.asarray(maps, np.float32).reshape(len(maps), -1)
    return x if extra is None else np.concatenate([x, np.asarray(extra, np.float32)], 1)


def sample(post, box, maps, n=1000, device=None, extra=None):
    """Posterior samples in theta space for each map: array (len(maps), n, D)."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.tensor(as_input(maps, extra), device=device)
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 64):
            zs = post.sample_batched((n,), x=x[i:i + 64], show_progress_bars=False)   # (n, b, D)
            out.append(np.transpose(zs.cpu().numpy(), (1, 0, 2)))
    return box.to_theta(np.concatenate(out))


def calibration(samples, theta_true, levels=(0.5, 0.68, 0.9, 0.95)):
    """Ranks (SBC) and empirical coverage of central credible intervals per parameter."""
    ranks = (samples < theta_true[:, None, :]).mean(1)                    # (N, D) in [0, 1]
    cov = {}
    for lev in levels:
        lo = np.quantile(samples, 0.5 - lev / 2, axis=1)
        hi = np.quantile(samples, 0.5 + lev / 2, axis=1)
        cov[lev] = ((theta_true >= lo) & (theta_true <= hi)).mean(0)
    return ranks, cov


def embed(est, maps, device=None, batch=512, extra=None):
    """Summary-network features of crack maps (N, out)."""
    device = device or next(est.parameters()).device
    net = est.embedding_net.eval()
    x = torch.tensor(as_input(maps, extra))
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(net(x[i:i + batch].to(device)).cpu().numpy())
    return np.concatenate(out)


def ood_score(train_feat, test_feat, obs_feat, k=10):
    """Distance of observations to their k nearest training summaries, as a percentile of the same
    distance for held-out simulations (values near 100 flag an observation the simulator does not
    produce, i.e. model misspecification)."""
    mu, sd = train_feat.mean(0), train_feat.std(0) + 1e-9
    tr = (train_feat - mu) / sd

    tn = (tr ** 2).sum(1)

    def knn(q):
        q = (q - mu) / sd
        out = []
        for i in range(0, len(q), 256):
            qb = q[i:i + 256]
            d2 = np.maximum((qb ** 2).sum(1)[:, None] + tn[None, :] - 2.0 * qb @ tr.T, 0.0)
            out.append(np.sqrt(np.partition(d2, k, axis=1)[:, :k]).mean(1))
        return np.concatenate(out)

    ref = knn(test_feat)
    obs = knn(obs_feat)
    return np.array([(ref < o).mean() * 100.0 for o in obs]), obs, ref
