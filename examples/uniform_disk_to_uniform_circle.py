import jax
import jax.numpy as jnp
from jax.scipy.special import i0e, i1e

from ott.neural.networks import potentials
import optax

from msinkhorn import ExpectileNeuralMOT
from msinkhorn.viz2d import plot_mtransport_validation_2d, animate_bass_martingale



def sample_uniform_disk(key, n):
    """Sample n points uniformly from the unit disk in R^2."""
    key_r, key_theta = jax.random.split(key)
    r = jnp.sqrt(jax.random.uniform(key_r, (n,)))
    theta = 2.0 * jnp.pi * jax.random.uniform(key_theta, (n,))
    return jnp.stack([r * jnp.cos(theta), r * jnp.sin(theta)], axis=1)


def sample_uniform_circle(key, n):
    """Sample n points uniformly from the unit circle in R^2."""
    theta = 2.0 * jnp.pi * jax.random.uniform(key, (n,))
    x = jnp.cos(theta)
    y = jnp.sin(theta)
    return jnp.stack([x, y], axis=1)


def h(r: jnp.ndarray) -> jnp.ndarray:
    """Mean resultant length function for the Bass measure."""
    z = 0.25 * r**2
    return jnp.sqrt(jnp.pi * z / 2.0) * (i0e(z) + i1e(z))

def make_h_table(r_max=30.0, n=10000, p=3.0):
    """Create lookup table for h and its inverse."""
    t = jnp.linspace(0.0, 1.0, n)
    r_grid = r_max * (t ** p)
    h_grid = jnp.maximum.accumulate(h(r_grid))
    return r_grid, h_grid

@jax.jit
def h_inv(u: jnp.ndarray, r_grid: jnp.ndarray, h_grid: jnp.ndarray) -> jnp.ndarray:
    """Inverse of the mean resultant length function using linear interpolation."""
    u = jnp.clip(u, 0.0, h_grid[-1] - 1e-12)
    return jnp.interp(u, h_grid, r_grid)

def sample_bass_measure(key, n, r_grid, h_grid):
    """Sample n points from the Bass measure associated to the uniform laws on the unit disk and circle."""
    key_u, key_theta = jax.random.split(key)
    r = jnp.sqrt(jax.random.uniform(key_u, (n,)))
    rho = h_inv(r, r_grid, h_grid)
    theta = 2.0 * jnp.pi * jax.random.uniform(key_theta, (n,))
    return jnp.stack([rho * jnp.cos(theta), rho * jnp.sin(theta)], axis=1)



def main():
    key = jax.random.PRNGKey(20251219)
    key, kmu0, kmu1, ksol = jax.random.split(key, 4)

    # training
    # nsamples = 20000
    # mu0 = sample_uniform_disk(kmu0, nsamples)
    # mu1 = sample_uniform_circle(kmu1, nsamples)
    
    # solver = ExpectileNeuralMOT(
    #     dim_data=2,
    #     neural_f=potentials.MLP(
    #         dim_hidden=[64, 64, 64, 64, 1],
    #         act_fn=jax.nn.silu,
    #     ),
    #     neural_g=potentials.MLP(
    #         dim_hidden=[64, 64, 64, 64, 1],
    #         act_fn=jax.nn.silu,
    #     ),
    #     neural_h=potentials.MLP(
    #         dim_hidden=[64, 64, 64, 64, 1],
    #         act_fn=jax.nn.silu,
    #         ),
    #     optimizer_f=optax.adam(5e-4),
    #     optimizer_g=optax.adam(5e-4),
    #     optimizer_h=optax.adam(5e-4),
    #     expectile=0.98, expectile_loss_coef=0.5,
    #     key=ksol,
    #     nsim = 512,
    # )

    # res = solver(
    #     num_train_iters=10, 
    #     batch_size=2048,
    #     num_iters_per_step={"ENOT": 4000, "gen": 200},
    #     train=(mu0, mu1),
    #     valid=(mu0, mu1),
    #     valid_batch_size=2048,
    #     valid_freqs={"ENOT": 500, "gen": 80, "train": 1},
    #     callbacks=[],
    # )
    # solver.save("./examples/uniform_disk_to_uniform_circle.ckpt")

    # load and validate
    solver_loaded = ExpectileNeuralMOT.load(
        "./examples/uniform_disk_to_uniform_circle.ckpt",
        neural_f=potentials.MLP(
            dim_hidden=[64, 64, 64, 64, 1],
            act_fn=jax.nn.silu,
        ),
        neural_g=potentials.MLP(
            dim_hidden=[64, 64, 64, 64, 1],
            act_fn=jax.nn.silu,
        ),
        neural_h=potentials.MLP(
            dim_hidden=[64, 64, 64, 64, 1],
            act_fn=jax.nn.silu,
        ),
        optimizer_f=optax.adam(5e-4),
        optimizer_g=optax.adam(5e-4),
        optimizer_h=optax.adam(5e-4),
    )
    res = solver_loaded.to_dual_potentials()

    key, kmu0, kmu1, kalpha, ksim = jax.random.split(key, 5)
    nsamples = 40000
    mu0 = sample_uniform_disk(kmu0, nsamples)
    mu1 = sample_uniform_circle(kmu1, nsamples)
    r_grid, h_grid = make_h_table(r_max=30.0, n=10000, p=3.0)
    alpha = sample_bass_measure(kalpha, nsamples, r_grid, h_grid)

    ax_lims = {
        "mu0": {"x": (-1,1), "y": (-1,1)},
        "mu1": {"x": (-1,1), "y": (-1,1)},
        "alpha": {"x": (-5,5), "y": (-5,5)},
        "alpha_conv": {"x": (-5,5), "y": (-5,5)},
    }
    plot_mtransport_validation_2d(
        res, 
        mu0, mu1, 
        alpha=alpha, 
        nsim=512, 
        batch_size=4000, # adjust based on memory constraints
        key=ksim, 
        bins=32, 
        usetex=True, 
        save_prefix="./examples/uniform_disk_to_uniform_circle", 
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
        save_to="./examples/uniform_disk_to_uniform_circle_3d.mp4",
        fps=60,
        view=(20, -50),
        hold_end=2.0,
        dpi=240,
    )

    animate_bass_martingale(
        M,
        mode="planar",
        figsize=(4,4),
        save_to="./examples/uniform_disk_to_uniform_circle_planar.mp4",
        fps=60,
        hold_end=2.0,
        dpi=240,
    )



if __name__ == "__main__":
    main()