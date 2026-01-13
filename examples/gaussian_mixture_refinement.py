import jax
import jax.numpy as jnp

from ott.neural.networks import potentials
import optax

from msinkhorn import ExpectileNeuralMOT
from msinkhorn.viz2d import plot_mtransport_validation_2d, animate_bass_martingale



#region normal mixtures setup
w1 = jnp.array([0.5, 0.3, 0.2])
mu0 = jnp.array([
    [0.0, 0.0],
    [2.5, 1.0],
    [-2.0, 2.5],
])
sigma0 = jnp.stack([
    jnp.array([[1.0, 0.4],
               [0.4, 0.7]]),
    jnp.array([[0.8, -0.2],
               [-0.2, 0.5]]),
    jnp.array([[0.6, 0.3],
               [0.3, 1.2]]),
])
L0 = jnp.linalg.cholesky(sigma0)

w2 = jnp.array([0.2, 0.5, 0.3])
s = jnp.array([
    [[1.8, 0.4],
     [-0.7, 1.6]],
    [[1.2, -1.5],
     [2.0, 0.3]],
    [[-1.0, 1.0],
     [0.5, 2.0]],
])
s1 = s[:, 0, :]
s2 = s[:, 1, :]
s3 = -(w2[0] * s1 + w2[1] * s2) / w2[2]
s = jnp.stack([s1, s2, s3], axis=1)
A0 = jnp.array([
    [1.0, 0.6, 1.2],
    [0.9, 1.3, 0.7],
    [1.1, 0.8, 1.5],
])
A1 = jnp.array([
    [0.7, 1.1, 0.5],
    [1.2, 0.6, 1.0],
    [0.8, 1.4, 0.9],
])
rho = jnp.array([
    [0.5, -0.3, 0.1],
    [0.7, 0.2, -0.4],
    [-0.6, 0.4, 0.3],
])
C11 = A0 ** 2
C22 = A1 ** 2
C12 = rho * A0 * A1
C = jnp.stack([
    jnp.stack([C11, C12], axis=-1),
    jnp.stack([C12, C22], axis=-1),
], axis=-2)
mu1 = mu0[:, None, :] + s
mu1_flat = mu1.reshape(-1, 2)
sigma1 = sigma0[:, None, :, :] + C
L1 = jnp.linalg.cholesky(sigma1)
L1_flat = L1.reshape(-1, 2, 2)


def sample_mu0(key, n):
    """Sample from the first mixture of Gaussians."""
    k0, k1 = jax.random.split(key)
    I = jax.random.categorical(k0, jnp.log(w1), shape=(n,))
    
    mean = mu0[I]
    chol = L0[I]
    z = jax.random.normal(k1, shape=(n, 2))
    eps = jnp.einsum("nij,nj->ni", chol, z)
    return mean + eps


def sample_mu1(key, n):
    """Sample from the second mixture of Gaussians."""
    k0, k1, k2 = jax.random.split(key, 3)
    I = jax.random.categorical(k0, jnp.log(w1), shape=(n,))
    J = jax.random.categorical(k1, jnp.log(w2), shape=(n,))
    flat_index = 3*I + J

    mean = mu1_flat[flat_index]
    chol = L1_flat[flat_index]
    z = jax.random.normal(k2, shape=(n, 2))
    eps = jnp.einsum("nij,nj->ni", chol, z)
    return mean + eps
#endregion



def main():
    key = jax.random.PRNGKey(2500)
    key, kmu0, kmu1, ksol = jax.random.split(key, 4)

    # training
    # nsamples = 20000
    # mu0 = sample_mu0(kmu0, nsamples)
    # mu1 = sample_mu1(kmu1, nsamples)

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
    #     num_train_iters=5, batch_size=1024,
    #     num_iters_per_step={"ENOT": 4000, "gen": 200},
    #     train=(mu0, mu1),
    #     valid=(mu0, mu1),
    #     valid_batch_size=1024,
    #     valid_freqs={"ENOT": 500, "gen": 80, "train": 1},
    #     callbacks=[],
    # )
    # solver.save("./examples/gaussian_mixture_refinement.ckpt")

    # load and validate
    solver_loaded = ExpectileNeuralMOT.load(
        "./examples/gaussian_mixture_refinement.ckpt",
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
    mu0 = sample_mu0(kmu0, nsamples)
    mu1 = sample_mu1(kmu1, nsamples)

    ax_lims = {
        "mu0": {"x": (-5,7), "y": (-5,7)},
        "mu1": {"x": (-8,11), "y": (-8,11)},
        "alpha": {"x": (-3,3), "y": (-2,4)},
        "alpha_conv": {"x": (-5,5), "y": (-4,6)},
    }
    plot_mtransport_validation_2d(
        res, 
        mu0, mu1, 
        nsim=512, 
        batch_size=4000, # adjust based on memory constraints
        key=ksim, 
        bins=32, 
        usetex=True, 
        save_prefix="./examples/gaussian_mixture_refinement", 
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
        save_to="./examples/gaussian_mixture_refinement_3d.mp4",
        fps=60,
        view=(20, -50),
        hold_end=2.0,
        dpi=240,
    )

    animate_bass_martingale(
        M,
        mode="planar",
        figsize=(4,4),
        save_to="./examples/gaussian_mixture_refinement_planar.mp4",
        fps=60,
        hold_end=2.0,
        dpi=240,
    )



if __name__ == "__main__":
    main()