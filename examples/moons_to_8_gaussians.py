import jax
import jax.numpy as jnp

from ott.neural.networks import potentials
import optax

from msinkhorn import ExpectileNeuralMOT
from msinkhorn.viz2d import plot_mtransport_validation_2d, animate_bass_martingale



def sample_8gaussians(key, nsamples, radius=2.0, sigma=0.05):
    """Sample from the 8-Gaussians dataset."""
    idx_key, noise_key = jax.random.split(key, 2)

    angles = jnp.linspace(0.0, 2.0 * jnp.pi, 9)[:-1]
    means = radius * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)

    idx = jax.random.randint(idx_key, (nsamples,), 0, 8)
    x = means[idx] + sigma * jax.random.normal(noise_key, (nsamples, 2))
    return x


def sample_moons(key, nsamples, shuffle=True, sigma=None, dtype=jnp.float32):
    """Sample from the two moons dataset (replicates sklearn.datasets.make_moons)."""
    if isinstance(nsamples, (tuple, list)):
        n_out = int(nsamples[0])
        n_in = int(nsamples[1])
    else:
        nsamples = int(nsamples)
        n_out = nsamples // 2
        n_in = nsamples - n_out

    key_out, key_in, key_noise, key_perm = jax.random.split(key, 4)

    t_out = jax.random.uniform(key_out, (n_out,), minval=0.0, maxval=jnp.pi, dtype=dtype)
    t_in  = jax.random.uniform(key_in,  (n_in,),  minval=0.0, maxval=jnp.pi, dtype=dtype)

    outer = jnp.stack([jnp.cos(t_out), jnp.sin(t_out)], axis=1)
    inner = jnp.stack([1.0 - jnp.cos(t_in), 1.0 - jnp.sin(t_in) - 0.5], axis=1)

    X = jnp.concatenate([outer, inner], axis=0).astype(dtype)
    y = jnp.concatenate(
        [jnp.zeros((n_out,), dtype=jnp.int32), jnp.ones((n_in,), dtype=jnp.int32)],
        axis=0,
    )

    if sigma is not None and float(sigma) != 0.0:
        X = X + (dtype(sigma) * jax.random.normal(key_noise, X.shape, dtype=dtype))

    if shuffle:
        perm = jax.random.permutation(key_perm, X.shape[0])
        X = X[perm]
        y = y[perm]

    return X, y



def main():
    key = jax.random.PRNGKey(20251212)
    key, kmu0, kmu1, ksol = jax.random.split(key, 4)

    # training
    # nsamples = 20000
    # mu0, _ = sample_moons(kmu0, nsamples, sigma=0.1)
    # mu0 -= jnp.array([0.5, 0.25])
    # mu1 = sample_8gaussians(kmu1, nsamples, sigma = 0.3, radius=1.75)

    # solver = ExpectileNeuralMOT(
    #     dim_data=2,
    #     neural_f=potentials.MLP(
    #         dim_hidden=[64, 64, 64, 1],
    #         act_fn=jax.nn.silu,
    #     ),
    #     neural_g=potentials.MLP(
    #         dim_hidden=[64, 64, 64, 1],
    #         act_fn=jax.nn.silu,
    #     ),
    #     neural_h=potentials.MLP(
    #         dim_hidden=[64, 64, 64, 1],
    #         act_fn=jax.nn.silu,
    #         ),
    #     optimizer_f=optax.adam(5e-4),
    #     optimizer_g=optax.adam(5e-4),
    #     optimizer_h=optax.adam(5e-4),
    #     expectile=0.98, expectile_loss_coef=0.5,
    #     key=ksol,
    #     nsim = 256,
    # )

    # res = solver(
    #     num_train_iters=10, batch_size=2048,
    #     num_iters_per_step={"ENOT": 4000, "gen": 200},
    #     train=(mu0, mu1),
    #     valid=(mu0, mu1),
    #     valid_batch_size=2048,
    #     valid_freqs={"ENOT": 500, "gen": 80, "train": 1},
    #     callbacks=[],
    # )
    # solver.save("./examples/moons_to_8_gaussians.ckpt")

    # load and validate
    solver_loaded = ExpectileNeuralMOT.load(
        "./examples/moons_to_8_gaussians.ckpt",
        neural_f=potentials.MLP(
            dim_hidden=[64, 64, 64, 1],
            act_fn=jax.nn.silu,
        ),
        neural_g=potentials.MLP(
            dim_hidden=[64, 64, 64, 1],
            act_fn=jax.nn.silu,
        ),
        neural_h=potentials.MLP(
            dim_hidden=[64, 64, 64, 1],
            act_fn=jax.nn.silu,
        ),
        optimizer_f=optax.adam(5e-4),
        optimizer_g=optax.adam(5e-4),
        optimizer_h=optax.adam(5e-4),
        )
    res = solver_loaded.to_dual_potentials()

    key, kmu0, kmu1, ksim = jax.random.split(key, 4)
    nsamples = 40000
    mu0, _ = sample_moons(kmu0, nsamples, sigma=0.1)
    mu0 -= jnp.array([0.5, 0.25])
    mu1 = sample_8gaussians(kmu1, nsamples, sigma = 0.3, radius=1.75)

    ax_lims = {
        "mu0": {"x": (-2,2), "y": (-2,2)},
        "mu1": {"x": (-3,3), "y": (-3,3)},
        "alpha": {"x": (-3,3), "y": (-2,2)},
        "alpha_conv": {"x": (-4,4), "y": (-4,4)},
    }
    plot_mtransport_validation_2d(
        res, 
        mu0, mu1, 
        nsim=512, 
        batch_size=4000, # adjust based on memory constraints
        key=ksim, 
        bins=32, 
        usetex=True, 
        save_prefix="./examples/moons_to_8_gaussians", 
        show=False, 
        nticks=2, 
        ax_lims=ax_lims
        )

    # Optional: simulate Bass martingale and create animations
    # Warning: memory-intensive (large [npaths, nsteps, d] arrays).
    key, ksim = jax.random.split(key)
    M_jax = res.sim_bass_martingale(
        x=mu0[:512],
        nsteps=256,
        ncond=512,
        key=ksim,
        batch_size=64,  # adjust based on memory constraints
    )
    M = jax.device_get(M_jax)

    animate_bass_martingale(
        M,
        mode="3d",
        figsize=(6,6),
        save_to="./examples/moons_to_8_gaussians_3d.mp4",
        fps=60,
        view=(20, -50),
        hold_end=2.0,
        dpi=240,
    )

    animate_bass_martingale(
        M,
        mode="planar",
        figsize=(4,4),
        save_to="./examples/moons_to_8_gaussians_planar.mp4",
        fps=60,
        hold_end=2.0,
        dpi=240,
    )



if __name__ == "__main__":
    main()