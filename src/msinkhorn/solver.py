from typing import Callable, Tuple, Dict, Optional

from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax import serialization
from flax import linen as nn
import optax

from ott.geometry import costs
from ott.neural.methods.expectile_neural_dual import ExpectileNeuralDual, PotentialModelWrapper, ENOTPotentials
from ott.neural.networks import potentials

from ._utils import DataSampler, LossTracker, microbatch

__version__ = "0.1.0"


@jtu.register_static
class NMOTPotentials():
    """
    Convenience wrapper around the neural MOT potentials.
    Notation:
        (mu0, mu1): source and target measures. Must be in convex order, irreducible and with finite second moments.
        alpha: the Bass measure.
        alpha_conv: the convolution of alpha with d-dimensional standard Gaussian.

    grad_vC (Callable[[jnp.ndarray], jnp.ndarray]): Gradient of the c-conjugate of the potential v, pushing from mu0 to alpha.
    enot_pot (ENOTPotentials): Expectile neural OT potentials between alpha_conv and mu1. enot_pot.f is the potential v* pushing from alpha_conv to mu1. enot_pot.g is the potential v pushing from mu1 to alpha_conv.
    """
    def __init__(self, grad_vC: Callable[[jnp.ndarray], jnp.ndarray], enot_pot: ENOTPotentials):
        self.grad_vC = grad_vC
        self.enot_pot = enot_pot

    def to_alpha(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Push samples from mu0 to samples from alpha.
        
        Args: 
            x (jnp.ndarray): Samples from mu0, shape [n, d].
        
        Returns:
            a (jnp.ndarray): Samples from alpha, shape [n, d].
        """
        return self.grad_vC(jnp.atleast_2d(x))
    
    def to_mu1(self, b: jnp.ndarray) -> jnp.ndarray:
        """
        Push samples from alpha + Z, where Z is standard Gaussian (d-dimensional), to samples from mu1.
        
        Args:
            b (jnp.ndarray): Samples from alpha + Z, shape [n, d].
        
        Returns:
            y (jnp.ndarray): Samples from mu1, shape [n, d].
        """
        return self.enot_pot.transport(jnp.atleast_2d(b), forward=True)
    
    def to_mu0(self, a: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        """
        Push samples from alpha to samples from mu0 using Monte Carlo convolution.

        Args:
            a (jnp.ndarray): Samples from alpha, shape [n, d].
            z (jnp.ndarray): Standard normally distributed random variables, shape [nsim, n, d].

        Returns:
            y (jnp.ndarray): Samples from mu0, shape [n, d].
        """
        b = jnp.atleast_2d(a)[None, :, :] + jnp.atleast_3d(z)   # [nsim, n, d]
        return jax.vmap(self.enot_pot.transport,in_axes=(0, None))(b, True).mean(axis=0)
    
    def to_alpha_conv(self, y: jnp.ndarray) -> jnp.ndarray:
        """
        Push samples from mu1 to samples from alpha + Z, where Z is standard Gaussian (d-dimensional).
        
        Args:
            y (jnp.ndarray): Samples from mu1, shape [n, d].
        
        Returns:
            b (jnp.ndarray): Samples from alpha + Z, shape [n, d].
        """
        return self.enot_pot.transport(jnp.atleast_2d(y), forward=False)

    def mtransport(self, x: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        """Push samples from mu0 to samples from mu1 via the Bass martingale transport.
        
        Args:
            x (jnp.ndarray): Samples from mu0, shape [n, d].
            z (jnp.ndarray): Standard normally distributed random variables, shape [n, d].

        Returns:
            y (jnp.ndarray): Samples from mu1, shape [n, d].
        """
        x_2d = jnp.atleast_2d(x)
        z_2d = jnp.atleast_2d(z)
        return self.to_mu1(self.to_alpha(x_2d) + z_2d)
    
    def sim_bass_martingale(self, key: jax.random.PRNGKey, x: jnp.ndarray, nsteps: int, ncond: int, batch_size: Optional[int] = None) -> jnp.ndarray:
        """Simulate paths of the Bass martingale transporting mu0 to mu1 via Monte Carlo Simulation:
        M_t ≈ E[∇v*(B_1) | B_t], where B_t = a + W_t, W_t is a standard Brownian motion, and a ~ alpha (the Bass measure).

        Args:
            key (jax.random.PRNGKey): JAX random key.
            x (jnp.ndarray): Samples from mu0, shape [npaths, d].
            nsteps (int): Number of time steps the martingale is simulated over.
            ncond (int): Number of conditional samples of B_1 given B_t to use in the Monte Carlo estimation of the conditional expectation.
            batch_size (Optional[int]): Batch size of how many paths to simulate at once. If None, simulates all paths at once.
        Returns:
            M (jnp.ndarray): Simulated paths of the Bass martingale transporting mu0 to mu1, shape [npaths, nsteps, d].
        """
        npaths, d = x.shape
        dtype = x.dtype

        def sim_fn(keys: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
            # simulate Brownian motion paths
            npaths = x.shape[0]
            alpha = self.to_alpha(x)                                    # [npaths, d]
        
            key_bm, key_y = jax.random.split(keys[0], 2)
            dt = 1.0 / (nsteps - 1)
            dW_t = jnp.sqrt(dt) * jax.random.normal(key_bm, (npaths, nsteps - 1, d), dtype=dtype)
            incr = jnp.concatenate([jnp.zeros((npaths, 1, d), dtype=dtype), jnp.cumsum(dW_t, axis=1)], axis=1)
            B_t = alpha[:, None, :] + incr                              # [npaths, nsteps, d]

            # simulate ncond B_1 given B_t
            t = jnp.linspace(0.0, 1.0, nsteps)
            sigma = jnp.sqrt(1.0 - t)
            sigma = sigma.at[-1].set(0.0)
            z = jax.random.normal(key_y, (npaths, nsteps, ncond, d), dtype=dtype)
            B_t = B_t[:, :, None, :]
            B_1 = B_t + sigma[None, :, None, None] * z

            # push B_1 to mu1
            y = self.to_mu1(B_1.reshape(-1, d))                         # [npaths * nsteps * ncond, d]
            return y.reshape(npaths, nsteps, ncond, d).mean(axis=2)     # [npaths, nsteps, d]
        
        fn = jax.jit(sim_fn)
        
        if batch_size is None or batch_size >= npaths:
            return fn(key, x)
        
        sim_bm_mb = microbatch(
            fn,
            batch_size=batch_size,
            in_axes=(0,0),
        )
        n_chunks = (npaths + batch_size - 1) // batch_size
        keys = jnp.repeat(jax.random.split(key, n_chunks), repeats=batch_size, axis=0)[:npaths]     # One unique key per microbatch chunk
        return sim_bm_mb(keys, x)
    


class ExpectileNeuralMOT():
    """Expectile-regularised Neural Martingale Optimal Transport.

    It solves the martingale Benamou-Brenier (mBB) problem which seeks a martingale closest to the Brownian motion connecting to marginals (mu0, mu1) in convex order. NOTE: This implementation assumes that mu0 and mu1 are in convex order, irreducible and have finite second moments.
    Its solution is the Bass martingale, i.e., stretched Brownian motion of the form M_t ≈ E[∇v*(B_1) | B_t], where B_t = a + W_t, W_t is a standard Brownian motion, and a ~ alpha (the Bass measure). '*' denotes the convex conjugate, i.e., v*(x) = sup_y x.y - v(y).
    v is called the Bass potential and can be obtained by solving the corresponding dual problem:
        inf_{v convex} E_{y~mu1}[ v(y) ] - E_{x~mu0}[ vC(x) ]
    where vC is the C-conjugate of v for the weak optimal transport cost:
        C(rho, x) = - MCov(rho, gamma) + 1/2 E_{z~rho}[ ||z||^2 ] if rho has mean x,
                    +infty otherwise.
    gamma is the standard Gaussian measure in R^d, and MCov(rho, gamma) = sup_{coupling} E_{(Z, G) ~ coupling}[ Z.G ]. vC can be computed as vC(x) = (E_{Z~gamma}[ v*(x + Z) ])*.
    
    This class implements the Martingale Sinkhorn algorithm as follows:
    - We parametrise v* and v using neural networks f and g, respectively, and use another neural network h to parametrise vC.
    - v* and v are trained using the ExpectileNeuralDual class from the ott-jax library. vC is trained such that its gradient minimises the Fenchel-Young gap with respect to E_{Z~gamma}[ v*(x + Z) ]. This expectation is approximated using Monte Carlo with nsim samples.
    NOTE: By the strong duality result of the mBB problem, it suffices to restrict the dual problem to potentials which grow quadratically at infinity. We parameterize vC = 0.5*||x||^2 - h(x), and similarly for v* and v.
    It is thus not recommended to use ICNN (input convex neural networks) architectures for f, g, and h in this implementation. 
    NOTE: The training does not perform affine normalisation of the potentials and only trains at the gradient level. As such, the dual loss is not guaranteed to be non-negative.

    Args:
        dim_data (int): Dimension of the data.
        neural_f (nn.Module): Neural network model that parametrises the potential v*, pushing from alpha + Z, where Z is standard Gaussian, to mu1.
        neural_g (nn.Module): Neural network model that parametrises the potential v, pushing from mu1 to alpha + Z, where Z is standard Gaussian.
        neural_h (nn.Module): Neural network model that parametrises the potential vC, pushing from mu0 to alpha.
        optimizer_f (optax.GradientTransformation): Optax optimizer for neural_f.
        optimizer_g (optax.GradientTransformation): Optax optimizer for neural_g.
        optimizer_h (optax.GradientTransformation): Optax optimizer for neural_h.
        expectile (float): Expectile level for the Expectile Neural OT solver. Must be in (0, 1).
        expectile_loss_coef (float): Expectile loss coefficient for the Expectile Neural OT solver. Must be positive.
        key (jax.random.PRNGKey): JAX random key.
        nsim (int): Number of Monte Carlo samples to approximate the convolution with Gaussian when training vC. Must be positive.
    """
    def __init__(
            self, 
            dim_data: int, 
            neural_f: nn.Module, 
            neural_g: nn.Module, 
            neural_h: nn.Module,
            optimizer_f: optax.GradientTransformation, 
            optimizer_g: optax.GradientTransformation,
            optimizer_h: optax.GradientTransformation,
            expectile: float, 
            expectile_loss_coef: float,
            key: jax.random.PRNGKey,
            nsim: int
            ):
        dim_data = int(dim_data)
        if dim_data <= 0:
            raise ValueError(f"dim_data must be positive integer, got {dim_data}")
        self.dim_data = dim_data

        nsim = int(nsim)
        if nsim <= 0:
            raise ValueError(f"nsim must be positive integer, got {nsim}")
        self.nsim = nsim

        self.key, key_ENOT, key_h = jax.random.split(key, 3)

        self.nds = ExpectileNeuralDual(
            dim_data=dim_data,
            neural_f=neural_f,
            neural_g=neural_g,
            optimizer_f=optimizer_f,
            optimizer_g=optimizer_g,
            cost_fn=costs.PNormP(2.0),  # Wasserstein-2 cost. Can be changed to other costs, e.g. Wasserstein-p, but needs corresponding changes in the _get_alpha_conv_rvs, _loss_gen, to_dual_potentials, and _dual_loss methods.
            num_train_iters=0,
            expectile=expectile,
            expectile_loss_coef=expectile_loss_coef,
            rng=key_ENOT
        )
        self.neural_h = PotentialModelWrapper(model=neural_h, is_potential=True)
        self.state_h = self.neural_h.create_train_state(key_h, optimizer_h, (dim_data,))

        self.train_gen_step = self._get_train_gen_step()
        self.valid_gen_step = self._get_valid_gen_step()
        self.alpha_conv_rvs = self._get_alpha_conv_rvs()
        self.valid_step = self._get_valid_step()

    #region serialisation / deserialisation
    def _get_state(self) -> Dict:
        """Get model + optimizer state as a serialisable dict."""
        return {
            "version": __version__,
            "dim_data": self.dim_data,
            "nsim": self.nsim,
            "key": self.key,
            "expectile": self.nds.expectile,
            "expectile_loss_coef": self.nds.expectile_loss_coef,
            "nds_state_f": self.nds.state_f,
            "nds_state_g": self.nds.state_g,
            "state_h": self.state_h,
        }
    
    def save(self, path: str) -> None:
        """Save model + optimizer state to a file."""
        state = self._get_state()
        bytes_out = serialization.to_bytes(state)
        with open(path, "wb") as f:
            f.write(bytes_out)

    @classmethod
    def load(
            cls,
            path: str,
            neural_f,
            neural_g,
            neural_h,
            optimizer_f,
            optimizer_g,
            optimizer_h,
            key_override: jax.random.PRNGKey = None,
        ) -> "ExpectileNeuralMOT":
        """Load model + optimizer state from a file."""

        with open(path, "rb") as f:
            bytes_in = f.read()

        dummy_key = jax.random.PRNGKey(0)
        dummy = cls(
            dim_data=1,
            neural_f=neural_f,
            neural_g=neural_g,
            neural_h=neural_h,
            optimizer_f=optimizer_f,
            optimizer_g=optimizer_g,
            optimizer_h=optimizer_h,
            expectile=0.5,
            expectile_loss_coef=1.0,
            key=dummy_key,
            nsim=1,
        )
        template = dummy._get_state()
        restored = serialization.from_bytes(template, bytes_in)

        if restored["version"] != __version__:
            print(F"Warning in msinkhorn.ExpectileNeuralMOT.load: loading model of version {restored['version']}, current version is {__version__}.")

        dim_data = int(restored["dim_data"])
        nsim = int(restored["nsim"])
        expectile = float(restored["expectile"])
        expectile_loss_coef = float(restored["expectile_loss_coef"])

        key = key_override if key_override is not None else restored["key"]

        obj = cls(
            dim_data=dim_data,
            neural_f=neural_f,
            neural_g=neural_g,
            neural_h=neural_h,
            optimizer_f=optimizer_f,
            optimizer_g=optimizer_g,
            optimizer_h=optimizer_h,
            expectile=expectile,
            expectile_loss_coef=expectile_loss_coef,
            key=key,
            nsim=nsim,
        )

        obj.nds.state_f = restored["nds_state_f"]
        obj.nds.state_g = restored["nds_state_g"]
        obj.state_h = restored["state_h"]

        return obj
    #endregion

    #region getters
    def _get_key(self) -> jax.Array:
        """Get a new JAX random key."""
        self.key, sub = jax.random.split(self.key)
        return sub
    
    def _get_alpha_conv_rvs(self):
        """Get function to sample from alpha_conv = alpha + Z, where Z is standard Gaussian."""
        def alpha_conv_rvs(state_h: potentials.PotentialTrainState, key: jax.Array, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
            x = batch["source"]
            grad_h_fn = jax.vmap(state_h.potential_gradient_fn(state_h.params))
            alpha = x - grad_h_fn(x)
            Z = jax.random.normal(key, x.shape)
            return alpha + Z
        return alpha_conv_rvs
    

    def _get_train_gen_step(self):
        """Get training step for the generator potential vC."""
        @jax.jit
        def train_gen_step(
            state_f: potentials.PotentialTrainState,
            state_g: potentials.PotentialTrainState,
            state_h: potentials.PotentialTrainState,
            batch: Dict[str, jnp.ndarray]
            ) -> Tuple[potentials.PotentialTrainState, jnp.ndarray]:
            loss_grad_fn = jax.value_and_grad(self._loss_gen, argnums=2, has_aux=False)
            loss, grads_h = loss_grad_fn(
                state_f.params,
                state_g.params,
                state_h.params,
                state_f.potential_gradient_fn,
                state_g.potential_value_fn,
                state_h.potential_gradient_fn,
                batch
            )
            return state_h.apply_gradients(grads=grads_h), loss
        return train_gen_step
    
    def _get_valid_gen_step(self):
        """Get validation step for the generator potential vC."""
        @jax.jit
        def valid_gen_step(
            state_f: potentials.PotentialTrainState,
            state_g: potentials.PotentialTrainState,
            state_h: potentials.PotentialTrainState,
            batch: Dict[str, jnp.ndarray]
            ) -> jnp.ndarray:
            loss = self._loss_gen(
                state_f.params,
                state_g.params,
                state_h.params,
                state_f.potential_gradient_fn,
                state_g.potential_value_fn,
                state_h.potential_gradient_fn,
                batch
            )
            return loss
        return valid_gen_step
    
    def _loss_gen(
            self, 
            params_f: jnp.ndarray,
            params_g: jnp.ndarray,
            params_h: jnp.ndarray,
            gradient_f: Callable[[jnp.ndarray], jnp.ndarray],
            g_value: Callable[[jnp.ndarray], jnp.ndarray],
            gradient_h: Callable[[jnp.ndarray], jnp.ndarray],
            batch: Dict[str, jnp.ndarray]
            ) -> jnp.ndarray:
        """Loss function for training the generator potential vC: the Fenchel-Young gap between vC and E_{Z~gamma}[ v*(x + Z) ]."""
        x = batch["source"]         # [batch_size, d]
        z = batch["Z"]              # [nsim, batch_size, d]

        grad_h_fn = jax.vmap(gradient_h(params_h))
        f_fn = ENOTPotentials(
            gradient_f(jax.lax.stop_gradient(params_f)), 
            g_value(jax.lax.stop_gradient(params_g)), 
            cost_fn=self.nds.cost_fn
            ).f
        f_fn = jax.vmap(jax.vmap(f_fn))

        a = x - grad_h_fn(x)                                # [batch_size, d]
        a_conv = a[None, :, :] + z                          # [nsim, batch_size, d]

        f_conv = f_fn(a_conv).mean(axis=0)                  # [batch_size, ]
        sqnorm_a = jnp.sum(a**2, axis=-1)
        vstar_conv = 0.5 * sqnorm_a - f_conv                # [batch_size, ]

        a_dot_x = jnp.sum(a * x, axis=1)                    # [batch_size, ]
        return (vstar_conv - a_dot_x).mean()
    

    def _get_valid_step(self):
        """Get validation step for the dual loss."""
        @jax.jit
        def valid_step(state_g: potentials.PotentialTrainState,
                       state_h: potentials.PotentialTrainState,
                       batch: Dict[str, jnp.ndarray]
                       ) -> jnp.ndarray:
            loss = self._dual_loss(
                state_g.params,
                state_h.params,
                state_g.potential_value_fn,
                state_h.potential_value_fn,
                batch
            )
            return loss
        return valid_step

    def _dual_loss(self,
                  params_g: jnp.ndarray,
                  params_h: jnp.ndarray,
                  g_value: Callable[[jnp.ndarray], jnp.ndarray],
                  h_value: Callable[[jnp.ndarray], jnp.ndarray],
                  batch: Dict[str, jnp.ndarray]
                  ) -> jnp.ndarray:
        """Dual loss function: E_{y~mu1}[ v(y) ] - E_{x~mu0}[ vC(x) ]."""
        x = batch["source"]                                     # [batch_size, d]
        sqnorm_x = jnp.sum(x**2, axis=-1, keepdims=True)
        y = batch["target"]                                     # [batch_size, d]
        sqnorm_y = jnp.sum(y**2, axis=-1, keepdims=True)

        g_value_fn = jax.vmap(g_value(params_g))
        v_y = 0.5 * sqnorm_y - g_value_fn(y)                    # [batch_size, ]
        h_value_fn = jax.vmap(h_value(params_h))
        vC_x = 0.5 * sqnorm_x - h_value_fn(x)                   # [batch_size, ]

        return v_y.mean() - vC_x.mean()

    def to_dual_potentials(self) -> NMOTPotentials:
        """Get wrapper around the trained dual potentials."""
        nds = self.nds
        enot_pot = nds.to_dual_potentials()

        grad_h_fn = jax.vmap(self.state_h.potential_gradient_fn(self.state_h.params))
        def grad_vC(x: jnp.ndarray) -> jnp.ndarray:
            return x - grad_h_fn(x)
        
        return NMOTPotentials(jax.jit(grad_vC), enot_pot)
    #endregion
    
    #region training
    def _clean_opt_states(self, state: potentials.PotentialTrainState) -> potentials.PotentialTrainState:
        """Clean optimizer states for the next Martingale Sinkhorn iteration."""
        return state.replace(opt_state = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), state.opt_state))


    def enot_step(self, state_f: potentials.PotentialTrainState, state_g: potentials.PotentialTrainState) -> Tuple[potentials.PotentialTrainState, potentials.PotentialTrainState]:
        """One training iteration for the ENOT potentials between alpha_conv and mu1."""
        do_valid = self.do_valid
        nds = self.nds
        loss_tracker = self.loss_tracker

        valid_freq = self.valid_freqs["ENOT"]
        train_batch, valid_batch = {}, {}
        pbar = tqdm(range(nds.num_train_iters), desc="ENOT step", leave=False)
        if do_valid:
            pbar.set_postfix({"ENOT_loss_f": None, "ENOT_loss_g": None})
        for step in pbar:
            if step % 2 == 0:
                train_batch["source"] = next(self.alpha_conv["train"])
                train_batch["target"] = next(self.mu1["train"])
                (state_f, state_g, _, _, _, _) = nds.train_step(state_f, state_g, train_batch)
            else:
                train_batch["target"] = next(self.alpha_conv["train"])
                train_batch["source"] = next(self.mu1["train"])
                (state_g, state_f, _, _, _, _) = nds.train_step(state_g, state_f, train_batch)

            if (step + 1) % valid_freq == 0:
                if do_valid:
                    valid_batch["source"] = next(self.alpha_conv["valid"])
                    valid_batch["target"] = next(self.mu1["valid"])
                    valid_loss_f, valid_loss_g, _ = nds.valid_step(state_f, state_g, valid_batch)
                    loss_tracker.update(valid_loss_f, key="ENOT_loss_f")
                    loss_tracker.update(valid_loss_g, key="ENOT_loss_g")
                    pbar.set_postfix({"ENOT_loss_f": f"{loss_tracker.monitor['ENOT_loss_f']:.5f}", "ENOT_loss_g": f"{loss_tracker.monitor['ENOT_loss_g']:.5f}"})
        
        state_f = self._clean_opt_states(state_f)
        state_g = self._clean_opt_states(state_g)
        return state_f, state_g


    def gen_step(self, state_h: potentials.PotentialTrainState) -> potentials.PotentialTrainState:
        """One training iteration for the generator potential vC."""
        do_valid = self.do_valid
        nds = self.nds
        state_f = nds.state_f
        state_g = nds.state_g
        loss_tracker = self.loss_tracker

        valid_freq = self.valid_freqs["gen"]
        train_batch, valid_batch = {}, {}
        pbar_gen = tqdm(range(self.num_iters_per_step_gen), desc="Generator step", leave=False)
        if do_valid:
            pbar_gen.set_postfix({"gen_loss": None})
        for step in pbar_gen:
            train_batch["source"] = next(self.mu0["train"])
            train_batch["Z"] = jax.random.normal(self._get_key(), (self.nsim, *train_batch["source"].shape))
            state_h, _ = self.train_gen_step(state_f, state_g, state_h, train_batch)

            if (step + 1) % valid_freq == 0:
                if do_valid:
                    valid_batch["source"] = next(self.mu0["valid"])
                    valid_batch["Z"] = jax.random.normal(self._get_key(), (self.nsim, *valid_batch["source"].shape))
                    valid_loss = self.valid_gen_step(state_f, state_g, state_h, valid_batch)
                    loss_tracker.update(valid_loss, key="gen_loss")
                    pbar_gen.set_postfix({"gen_loss": f"{loss_tracker.monitor['gen_loss']:.5f}"})

        state_h = self._clean_opt_states(state_h)
        return state_h


    def train_step(self) -> None:
        """One iteration of the Martingale Sinkhorn algorithm."""
        # update ENOT potentials
        nds = self.nds
        nds.state_f, nds.state_g = self.enot_step(nds.state_f, nds.state_g)
        
        # update h
        self.state_h = self.gen_step(self.state_h)
        
        self.loss_tracker.reset()

        # update alpha_conv
        train_batch = {"source": self.mu0["train"].data}
        self.alpha_conv["train"].data = self.alpha_conv_rvs(self.state_h, self._get_key(), train_batch)
        do_valid = self.do_valid
        if do_valid:
            valid_batch = {"source": self.mu0["valid"].data}
            self.alpha_conv["valid"].data = self.alpha_conv_rvs(self.state_h, self._get_key(), valid_batch)


    def _validate_call_args(
            self,
            train: Tuple[jnp.ndarray, jnp.ndarray],
            valid: Tuple[jnp.ndarray, jnp.ndarray] | None,
            num_train_iters: int,
            batch_size: int,
            valid_batch_size: int | None,
            num_iters_per_step: Dict[str, int],
            valid_freqs: Dict[str, int],
            alpha0_train: jnp.ndarray | None,
            alpha0_valid: jnp.ndarray | None,
        ) -> bool:
        """Validate the arguments provided to __call__.

        Args:
            train: Training data (x_train, y_train).
            valid: Validation data (x_valid, y_valid) or None.
            num_train_iters: Number of Martingale Sinkhorn iterations to perform.
            batch_size: Batch size for training data.
            valid_batch_size: Batch size for validation data.
            num_iters_per_step: Number of iterations per Martingale Sinkhorn step for the ENOT and generator trainings.
            valid_freqs: Validation frequencies for ENOT, generator, and training dual loss.
            alpha0_train: Initial Bass measure samples for training data.
            alpha0_valid: Initial Bass measure samples for validation data.
        Returns:
            bool: True if validation can be performed, False otherwise.
        """
        if num_train_iters <= 0:
            raise ValueError(f"num_train_iters must be > 0, got {num_train_iters}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if valid_batch_size is not None and valid_batch_size <= 0:
            raise ValueError(f"valid_batch_size must be > 0, got {valid_batch_size}")

        if not isinstance(train, tuple) or len(train) != 2:
            raise TypeError("train must be a tuple (x_train, y_train)")
        x_train, y_train = train
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"train[0] and train[1] must have same first dimension; "
                f"got {x_train.shape[0]} and {y_train.shape[0]}"
            )
        if alpha0_train is not None and alpha0_train.shape[0] != x_train.shape[0]:
            raise ValueError(
                f"alpha0_train must have same first dimension as train[0]; "
                f"got {alpha0_train.shape[0]} and {x_train.shape[0]}"
            )

        for key in ("ENOT", "gen"):
            if key not in num_iters_per_step:
                raise KeyError(f"num_iters_per_step must contain key '{key}'")
            if num_iters_per_step[key] <= 0:
                raise ValueError(f"num_iters_per_step['{key}'] must be > 0")
            
        for key in ("ENOT", "gen", "train"):
            if key not in valid_freqs:
                raise KeyError(f"valid_freqs must contain key '{key}'")
            if valid_freqs[key] <= 0:
                raise ValueError(f"valid_freqs['{key}'] must be > 0")

        if valid is None:
            return False

        if not isinstance(valid, tuple) or len(valid) != 2:
            raise TypeError("valid must be a tuple (x_valid, y_valid) when provided")

        x_valid, y_valid = valid
        if x_valid.shape[0] != y_valid.shape[0]:
            raise ValueError(
                f"valid[0] and valid[1] must have same first dimension; "
                f"got {x_valid.shape[0]} and {y_valid.shape[0]}"
            )

        if valid_batch_size is None:
            raise ValueError(
                "valid_batch_size must be specified (or inferable) when validation data is given"
            )

        if alpha0_valid is not None and alpha0_valid.shape[0] != x_valid.shape[0]:
            raise ValueError(
                f"alpha0_valid must have same first dimension as valid[0]; "
                f"got {alpha0_valid.shape[0]} and {x_valid.shape[0]}"
            )

        return True


    def __call__(
            self,
            num_train_iters: int, 
            batch_size: int,
            num_iters_per_step: Dict[str, int],
            train: Tuple[jnp.ndarray, jnp.ndarray], 
            alpha0_train: Optional[jnp.ndarray] = None,
            valid: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None, 
            alpha0_valid: Optional[jnp.ndarray] = None,
            valid_batch_size: Optional[int] = None, 
            valid_freqs: Optional[Dict[str, int]] = {
                "ENOT": 500,
                "gen": 50,
                "train": 1},
            loss_tracker: LossTracker = LossTracker(
                monitor={
                    "ENOT_loss_f": None, "ENOT_loss_g": None,
                    "gen_loss": None,
                    "dual_loss": None}, 
                    alpha={ 
                        "ENOT_loss_f": 0.3, "ENOT_loss_g": 0.3,
                        "gen_loss": 0.3,
                        "dual_loss": 0.9}),
            callbacks: Optional[list] = []
            ) -> NMOTPotentials:
        """Train the Expectile Neural Martingale Optimal Transport potentials.
        
        Args:
            num_train_iters (int): Number of Martingale Sinkhorn iterations to perform.
            batch_size (int): Batch size for training data.
            num_iters_per_step (Dict[str, int]): Number of iterations per Martingale Sinkhorn step for the ENOT and generator trainings. NOTE: Requires keys "ENOT" and "gen".
            train (Tuple[jnp.ndarray, jnp.ndarray]): Training data (x_train, y_train), shapes: [n_x, d], [n_y, d].
            alpha0_train (Optional[jnp.ndarray]): Initial Bass measure samples for training data. Defaults to None, in which case x_train is used. Shape: [n_x, d].
            valid (Optional[Tuple[jnp.ndarray, jnp.ndarray]]): Validation data (x_valid, y_valid) or None.
            alpha0_valid (Optional[jnp.ndarray]): Initial Bass measure samples for validation data. Defaults to None, in which case x_valid is used. Shape: [n_x, d].
            valid_batch_size (Optional[int]): Batch size for validation data.
            valid_freqs (Optional[Dict[str, int]]): Validation frequencies for ENOT, generator, and training dual loss. NOTE: Requires keys "ENOT", "gen", and "train".
            loss_tracker (LossTracker): LossTracker instance to track losses during training. NOTE: Requires keys "ENOT_loss_f", "ENOT_loss_g", "gen_loss", and "dual_loss" in monitor and alpha.
            callbacks (Optional[list]): List of callback functions to be called during training.
        Returns:
            NMOTPotentials: Wrapper around the trained dual potentials.
        """
        if valid_batch_size is None:
            valid_batch_size = batch_size

        self.do_valid = self._validate_call_args(
            train,
            valid,
            num_train_iters,
            batch_size,
            valid_batch_size,
            num_iters_per_step,
            valid_freqs,
            alpha0_train,
            alpha0_valid,
        )

        # handle data
        self.mu0 = {
            "train": DataSampler(train[0], batch_size, self._get_key()),
            "valid": DataSampler(valid[0], valid_batch_size, self._get_key()) if self.do_valid else None,
        }
        self.mu1 = {
            "train": DataSampler(train[1], batch_size, self._get_key()),
            "valid": DataSampler(valid[1], valid_batch_size, self._get_key()) if self.do_valid else None,
        }

        if alpha0_train is None:
            alpha0_train = self.mu0["train"].data
        alpha_conv_train = alpha0_train + jax.random.normal(self._get_key(), alpha0_train.shape)

        if self.do_valid:
            if alpha0_valid is None:
                alpha0_valid = self.mu0["valid"].data
            alpha_conv_valid = alpha0_valid + jax.random.normal(self._get_key(), alpha0_valid.shape)
        else:
            alpha_conv_valid = None

        self.alpha_conv = {
            "train": DataSampler(alpha_conv_train, batch_size, self._get_key()),
            "valid": DataSampler(alpha_conv_valid, valid_batch_size, self._get_key()) if self.do_valid else None,
        }

        self.nds.num_train_iters = num_iters_per_step["ENOT"]
        self.num_iters_per_step_gen = num_iters_per_step["gen"]

        self.loss_tracker = loss_tracker
        self.valid_freqs = valid_freqs

        # training loop
        nds = self.nds
        loss_tracker = self.loss_tracker
        valid_freq = valid_freqs["train"]
        
        valid_batch = {}
        pbar = tqdm(range(num_train_iters), desc="Training")
        if self.do_valid:
            pbar.set_postfix({"dual_loss": None})
        for step in pbar:
            self.train_step()

            if (step + 1) % valid_freq == 0:
                if self.do_valid:
                    valid_batch["source"] = next(self.mu0["valid"])
                    valid_batch["target"] = next(self.mu1["valid"])
                    dual_loss = self.valid_step(nds.state_g, self.state_h, valid_batch)
                    loss_tracker.update(dual_loss, key="dual_loss")

                    pbar.set_postfix({"dual_loss": f"{loss_tracker.monitor["dual_loss"]:.5f}"})
                
                for cb in callbacks:
                    stop, nds.state_f, nds.state_g, self.state_h = cb(loss_tracker.monitor["dual_loss"], nds.state_f, nds.state_g, self.state_h)
                    if stop:
                        break
                
        return self.to_dual_potentials()
    #endregion