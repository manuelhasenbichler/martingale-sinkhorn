from typing import Callable, Tuple, Dict, Optional, Literal

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgba

from ._utils import microbatch

#region 2D heatmap visualization
def _set_style(usetex: Optional[bool] = False):
    """Set matplotlib style for pretty plots."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.4,
        "xtick.major.width": 1.4,
        "ytick.major.width": 1.4,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "image.cmap": "viridis",
        "text.usetex": usetex,
        "font.family": "serif" if usetex else "sans-serif"
    })


def _pretty_heat_axes(ax, title, xlabel, ylabel=None):
    """Apply pretty formatting to heatmap axes."""
    ax.set_title(title, pad=8)
    ax.set_xlabel(xlabel, labelpad=3.5)
    if ylabel is not None:
        ax.set_ylabel(ylabel, labelpad=4)
    ax.set_aspect("equal")
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        width=1.4,
        length=5,
        top=True,
        right=True,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)


def _plot_pair_heatmap(
        data_true,
        data_approx,
        title_true: str,
        title_approx: str,
        bins: Optional[int] = 64,
        usetex: Optional[bool] = False,
        fontsize: Optional[float] = None,
        max_step: Optional[float] = 0.5,
        nticks: Optional[int] = 2,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
    ):
    """Plot pair of heatmaps and their difference."""
    _set_style(usetex)

    if fontsize is not None:
        plt.rcParams.update({
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
        })

    data_true = np.asarray(data_true)
    data_approx = np.asarray(data_approx)

    fig = plt.figure(figsize=(7.5, 5))

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.0,
        hspace=0.45,
    )

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax_diff = fig.add_subplot(gs[1, :])

    if xlim is None:
        x_all = np.concatenate([data_true[:, 0], data_approx[:, 0]])
        xlim = (float(x_all.min()), float(x_all.max()))
    if ylim is None:
        y_all = np.concatenate([data_true[:, 1], data_approx[:, 1]])
        ylim = (float(y_all.min()), float(y_all.max()))
    
    def _snap_limits(lim, step: float):
        lo, hi = float(lim[0]), float(lim[1])
        if step <= 0:
            return (lo, hi)
        lo_s = np.floor(lo / step) * step
        hi_s = np.ceil(hi / step) * step

        if np.isclose(hi_s, lo_s):
            hi_s = lo_s + step
        return (float(lo_s), float(hi_s))

    xlim = _snap_limits(xlim, max_step)
    ylim = _snap_limits(ylim, max_step)
    
    xedges = np.linspace(xlim[0], xlim[1], bins + 1)
    yedges = np.linspace(ylim[0], ylim[1], bins + 1)

    H_true, _, _ = np.histogram2d(
        data_true[:, 0],
        data_true[:, 1],
        bins=[xedges, yedges],
        density=True,
    )
    H_true = H_true.T

    H_approx, _, _ = np.histogram2d(
        data_approx[:, 0],
        data_approx[:, 1],
        bins=[xedges, yedges],
        density=True,
    )
    H_approx = H_approx.T

    vmin, vmax = H_true.min(), H_true.max()

    def format_tick(t: float) -> str:
        if np.isclose(t, round(t)):
            return str(int(round(t)))
        if np.isclose(t * 2, round(t * 2)):
            return f"{t:.1f}".rstrip("0").rstrip(".")
        return f"{t:.2f}".rstrip("0").rstrip(".")

    def make_ticks(a_min, a_max, step, nticks):
        low = np.floor(a_min / step) * step
        high = np.ceil(a_max / step) * step
        candidates = np.arange(low, high + 1e-9, step)
        if len(candidates) <= nticks:
            ticks = candidates
        else:
            idx = np.linspace(0, len(candidates) - 1, nticks)
            idx = np.round(idx).astype(int)
            ticks = np.unique(candidates[idx])
        labels = [format_tick(t) for t in ticks]
        return ticks, labels

    xticks, xticklabels = make_ticks(xlim[0], xlim[1], max_step, nticks)
    yticks, yticklabels = make_ticks(ylim[0], ylim[1], max_step, nticks)

    im0 = ax0.pcolormesh(
        xedges,
        yedges,
        H_true,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    _pretty_heat_axes(ax0, title_true, xlabel=r"$x_1$", ylabel=r"$x_2$")
    ax0.set_xlim(xedges[0], xedges[-1])
    ax0.set_ylim(yedges[0], yedges[-1])
    ax0.set_xticks(xticks)
    ax0.set_yticks(yticks)
    ax0.set_xticklabels(xticklabels)
    ax0.set_yticklabels(yticklabels)

    im1 = ax1.pcolormesh(
        xedges,
        yedges,
        H_approx,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    _pretty_heat_axes(ax1, title_approx, xlabel=r"$x_1$", ylabel=r"$x_2$")
    ax1.set_xlim(xedges[0], xedges[-1])
    ax1.set_ylim(yedges[0], yedges[-1])
    ax1.set_xticks(xticks)
    ax1.set_yticks(yticks)
    ax1.set_xticklabels(xticklabels)
    ax1.set_yticklabels(yticklabels)

    H_diff = H_true - H_approx
    vmax_diff = np.abs(H_diff).max()
    vmin_diff = -vmax_diff

    im_diff = ax_diff.pcolormesh(
        xedges,
        yedges,
        H_diff,
        shading="auto",
        vmin=vmin_diff,
        vmax=vmax_diff,
        cmap="coolwarm",
        rasterized=True,
    )
    _pretty_heat_axes(
        ax_diff,
        title=title_true + r" $-$ " + title_approx,
        xlabel=r"$x_1$",
        ylabel=r"$x_2$",
    )
    ax_diff.set_xlim(xedges[0], xedges[-1])
    ax_diff.set_ylim(yedges[0], yedges[-1])
    ax_diff.set_xticks(xticks)
    ax_diff.set_yticks(yticks)
    ax_diff.set_xticklabels(xticklabels)
    ax_diff.set_yticklabels(yticklabels)

    tick_fs = fontsize if fontsize is not None else 10

    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    cbar0 = fig.colorbar(im0, cax=cax0)
    cbar0.ax.tick_params(width=1.2, length=4, labelsize=tick_fs)

    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.ax.tick_params(width=1.2, length=4, labelsize=tick_fs)

    dividerd = make_axes_locatable(ax_diff)
    caxd = dividerd.append_axes("right", size="5%", pad=0.05)
    cbar_bot = fig.colorbar(im_diff, cax=caxd)
    cbar_bot.ax.tick_params(width=1.2, length=4, labelsize=tick_fs)

    return fig, (ax0, ax1)


def _plot_single_heatmap(
        data,
        title: str,
        xlabel: str,
        ylabel: str,
        bins: Optional[int] = 64,
        usetex: Optional[bool] = False,
        fontsize: Optional[float] = None,
        max_step: Optional[float] = 0.5,
        nticks: Optional[int] = 2,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
    ):
    """Plot single heatmap."""
    _set_style(usetex)

    if fontsize is not None:
        plt.rcParams.update({
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
        })

    data = np.asarray(data)

    fig, ax = plt.subplots(1, 1, figsize=(3.55, 2.40))

    if xlim is None:
        x_all = data[:, 0]
        xlim = (float(x_all.min()), float(x_all.max()))
    if ylim is None:
        y_all = data[:, 1]
        ylim = (float(y_all.min()), float(y_all.max()))

    def _snap_limits(lim, step: float):
        lo, hi = float(lim[0]), float(lim[1])
        if step <= 0:
            return (lo, hi)
        lo_s = np.floor(lo / step) * step
        hi_s = np.ceil(hi / step) * step

        if np.isclose(hi_s, lo_s):
            hi_s = lo_s + step
        return (float(lo_s), float(hi_s))

    xlim = _snap_limits(xlim, max_step)
    ylim = _snap_limits(ylim, max_step)
    
    xedges = np.linspace(xlim[0], xlim[1], bins + 1)
    yedges = np.linspace(ylim[0], ylim[1], bins + 1)

    H, _, _ = np.histogram2d(
        data[:, 0],
        data[:, 1],
        bins=[xedges, yedges],
        density=True,
    )
    H = H.T

    def format_tick(t: float) -> str:
        if np.isclose(t, round(t)):
            return str(int(round(t)))
        if np.isclose(t * 2, round(t * 2)):
            return f"{t:.1f}".rstrip("0").rstrip(".")
        return f"{t:.2f}".rstrip("0").rstrip(".")

    def make_ticks(a_min, a_max, step, nticks):
        low = np.floor(a_min / step) * step
        high = np.ceil(a_max / step) * step
        candidates = np.arange(low, high + 1e-9, step)
        if len(candidates) <= nticks:
            ticks = candidates
        else:
            idx = np.linspace(0, len(candidates) - 1, nticks)
            idx = np.round(idx).astype(int)
            ticks = np.unique(candidates[idx])
        labels = [format_tick(t) for t in ticks]
        return ticks, labels

    xticks, xticklabels = make_ticks(xlim[0], xlim[1], max_step, nticks)
    yticks, yticklabels = make_ticks(ylim[0], ylim[1], max_step, nticks)

    im = ax.pcolormesh(
        xedges,
        yedges,
        H,
        shading="auto",
        rasterized=True,
    )
    _pretty_heat_axes(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    ax.set_xlim(xedges[0], xedges[-1])
    ax.set_ylim(yedges[0], yedges[-1])
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    tick_fs = fontsize if fontsize is not None else 10
    cbar.ax.tick_params(labelsize=tick_fs, width=1.2, length=4)

    return fig, ax


def plot_mtransport_validation_2d(
        pot,
        mu0: jnp.ndarray,
        mu1: jnp.ndarray,
        nsim: int,
        key: jax.random.PRNGKey,
        batch_size: Optional[int] = None,
        alpha: Optional[jnp.ndarray] = None,
        bins: Optional[int] = 64,
        show: Optional[bool] = True,
        save_prefix: Optional[str] = None,
        export_prefix: Optional[str] = None,
        usetex: Optional[bool] = False,
        fontsize: Optional[float] = None,
        max_step: Optional[float] = 0.5,
        nticks: Optional[int] = 2,
        ax_lims: Optional[Dict[str, Tuple[float, float]]] = {
            "mu1": {"x": None, "y": None},
            "alpha": {"x": None, "y": None},
            "alpha_conv": {"x": None, "y": None},
            "mu0": {"x": None, "y": None}
            }
    ) -> Tuple[plt.Figure, plt.Axes, plt.Figure, plt.Axes, plt.Figure, plt.Axes]:
    """
    Plot validation results for 2D martingale optimal transport.

    Args:
        pot (NMOTPotentials): Wrapper for the learned potentials.
        mu0 (jnp.ndarray): Samples from the source measure [n_x, 2].
        mu1 (jnp.ndarray): Samples from the target measure [n_y, 2].
        nsim (int): Number of Monte Carlo samples for computing the transport from alpha to mu0.
        batch_size (Optional[int]): Batch size for potential evaluations. If None, evaluate all at once.
        key (jax.random.PRNGKey): JAX random key.
        alpha (Optional[jnp.ndarray]): Samples from the true Bass measure [n_x, 2] to compare against. If None, only the approximate alpha is plotted.
        bins (Optional[int]): Number of bins for the heatmaps.
        show (Optional[bool]): Whether to show the plots.
        save_prefix (Optional[str]): If provided, prefix for saving the plots as PDF files.
        export_prefix (Optional[str]): If provided, prefix for exporting the data as CSV files.
        usetex (Optional[bool]): Whether to use LaTeX for rendering text.
        fontsize (Optional[float]): Font size for the plots. If None, default size is used.
        max_step (Optional[float]): Maximum step size for the axes ticks.
        nticks (Optional[int]): Number of ticks on each axis.
        ax_lims (Optional[Dict[str, Tuple[float, float]]]): Axis limits for the plots. Keys are "mu1", "alpha", "alpha_conv", "mu0", each mapping to a dict with keys "x" and "y" for the respective axis limits.

    Returns:
        Tuple[plt.Figure, plt.Axes, plt.Figure, plt.Axes, plt.Figure, plt.Axes]: Figures and axes for the three plots (mu1, alpha, alpha_conv).
    """
    _set_style(usetex)

    # mu0 -> mu1
    key, sub = jax.random.split(key)
    Z = jax.random.normal(sub, mu0.shape)
    if batch_size is None:
        mu1_approx = pot.mtransport(mu0, Z)
    else:
        mtransport_mb = microbatch(
            pot.mtransport,
            batch_size=batch_size,
            in_axes=(0, 0),
        )
        mu1_approx = mtransport_mb(mu0, Z)

    ax_lims_mu1 = ax_lims.get("mu1", {"x": None, "y": None})
    xlim = ax_lims_mu1.get("x", None)
    ylim = ax_lims_mu1.get("y", None)
    fig1, ax1 = _plot_pair_heatmap(
        mu1,
        mu1_approx,
        title_true=r"$\mu_1$",
        title_approx=r"$(\nabla f_{\theta_1})_\# (\alpha_{\theta_2} * \gamma)$",
        bins=bins,
        usetex=usetex,
        fontsize=fontsize,
        max_step=max_step,
        nticks=nticks,
        xlim=xlim,
        ylim=ylim,
    )
    if save_prefix is not None:
        fig1.savefig(f"{save_prefix}_mu1_transport.pdf",
                        bbox_inches="tight")
    if show:
        plt.show()

    # alpha
    if alpha is not None:
        alpha = jnp.atleast_2d(alpha)
        if batch_size is None:
            alpha_approx = pot.to_alpha(mu0)
        else:
            to_alpha_mb = microbatch(
                pot.to_alpha,
                batch_size=batch_size,
                in_axes=0,
            )
            alpha_approx = to_alpha_mb(mu0)

        ax_lims_alpha = ax_lims.get("alpha", {"x": None, "y": None})
        xlim = ax_lims_alpha.get("x", None)
        ylim = ax_lims_alpha.get("y", None)
        fig2, ax2 = _plot_pair_heatmap(
            alpha,
            alpha_approx,
            title_true=r"$\alpha$",
            title_approx=r"$(\nabla h_{\theta_2})_\# \mu_0$",
            bins=bins,
            usetex=usetex,
            fontsize=fontsize,
            max_step=max_step,
            nticks=nticks,
            xlim=xlim,
            ylim=ylim,
        )
    else:
        if batch_size is None:
            alpha_approx = pot.to_alpha(mu0)
        else:
            to_alpha_mb = microbatch(
                pot.to_alpha,
                batch_size=batch_size,
                in_axes=0,
            )
            alpha_approx = to_alpha_mb(mu0)

        ax_lims_alpha = ax_lims.get("alpha", {"x": None, "y": None})
        xlim = ax_lims_alpha.get("x", None)
        ylim = ax_lims_alpha.get("y", None)
        fig2, ax2 = _plot_single_heatmap(
            alpha_approx,
            title=r"$(\nabla h_{\theta_2})_\# \mu_0$",
            xlabel=r"$\alpha_1$",
            ylabel=r"$\alpha_2$",
            bins=bins,
            usetex=usetex,
            fontsize=fontsize,
            max_step=max_step,
            nticks=nticks,
            xlim=xlim,
            ylim=ylim,
        )

    if save_prefix is not None:
        fig2.savefig(f"{save_prefix}_alpha.pdf", bbox_inches="tight")
    if show:
        plt.show()

    # mu1 -> alpha + Z
    key, sub = jax.random.split(key)
    alpha_conv = alpha_approx + jax.random.normal(sub, alpha_approx.shape)
    if batch_size is None:
        alpha_conv_approx = pot.to_alpha_conv(mu1)
    else:
        to_alpha_conv_mb = microbatch(
            pot.to_alpha_conv,
            batch_size=batch_size,
            in_axes=0,
        )
        alpha_conv_approx = to_alpha_conv_mb(mu1)

    ax_lims_alpha_conv = ax_lims.get("alpha_conv", {"x": None, "y": None})
    xlim = ax_lims_alpha_conv.get("x", None)
    ylim = ax_lims_alpha_conv.get("y", None)
    fig3, ax3 = _plot_pair_heatmap(
        alpha_conv,
        alpha_conv_approx,
        title_true=r"$\alpha * \gamma$",
        title_approx=r"$(\nabla g_{\theta_1})_\# \mu_1$",
        bins=bins,
        usetex=usetex,
        fontsize=fontsize,
        max_step=max_step,
        nticks=nticks,
        xlim=xlim,
        ylim=ylim,
    )
    if save_prefix is not None:
        fig3.savefig(f"{save_prefix}_mu1_to_alpha_conv.pdf", bbox_inches="tight")
    if show:
        plt.show()

    # alpha -> mu0
    key, sub = jax.random.split(key)
    z = jax.random.normal(sub, shape=(nsim, *alpha_approx.shape))
    if batch_size is None:
        mu0_approx = pot.to_mu0(alpha_approx, z)
    else:
        to_mu0_mb = microbatch(
            pot.to_mu0,
            batch_size=batch_size,
            in_axes=(0, 1),
        )
        mu0_approx = to_mu0_mb(alpha_approx, z)

    ax_lims_mu0 = ax_lims.get("mu0", {"x": None, "y": None})
    xlim = ax_lims_mu0.get("x", None)
    ylim = ax_lims_mu0.get("y", None)
    fig4, ax4 = _plot_pair_heatmap(
        mu0,
        mu0_approx,
        title_true=r"$\mu_0$",
        title_approx=r"$(\nabla (f_{\theta_1} * \gamma))_\# \alpha_{\theta_2}$",
        bins=bins,
        usetex=usetex,
        fontsize=fontsize,
        max_step=max_step,
        nticks=nticks,
        xlim=xlim,
        ylim=ylim,
    )
    if save_prefix is not None:
        fig4.savefig(f"{save_prefix}_alpha_to_mu0.pdf",
                        bbox_inches="tight")
    if show:
        plt.show()

    if export_prefix is not None:
        def _save(name, arr):
            arr = np.asarray(arr)
            np.savetxt(
                f"{export_prefix}_{name}.csv",
                arr,
                delimiter=",",
                header=r"$x_1,x_2$",
                comments="",
            )

        _save("mu0", mu0)
        _save("mu1", mu1)
        if alpha is not None:
            _save("alpha", alpha)
        _save("mu1_approx", mu1_approx)
        _save("alpha_approx", alpha_approx)
        _save("alpha_conv", alpha_conv)
        _save("alpha_conv_approx", alpha_conv_approx)
        _save("mu0_approx", mu0_approx)

    return {
        "mu1_transport": (fig1, ax1),
        "alpha": (fig2, ax2),
        "mu1_to_alpha_conv": (fig3, ax3),
        "alpha_to_mu0": (fig4, ax4),
    }
#endregion



#region animation
def animate_bass_martingale(
        paths,
        mode: Literal["planar", "3d"] = "3d",
        fps: Optional[int] = 30,
        max_frames: Optional[int] = 260,
        frame_stride: Optional[str] = "auto",
        figsize: Optional[tuple] = (6,6),
        view: Optional[tuple] = (18, -60),
        time_stretch: Optional[float] = 5.0,
        point_size: Optional[float] = 10.0,
        start_color: Optional[str] = "#1F77B4",
        thread_color: Optional[str] = "#2A9D8F",
        end_color: Optional[str] = "#2B2B2B",
        thread_alpha: Optional[float] = 0.45,
        thread_width: Optional[float] = 0.10,
        max_threads: Optional[int] = 200,
        hide_axes: Optional[bool] = True,
        ylim: Optional[str] = "auto",
        zlim: Optional[str] = "auto",
        lim_pad: Optional[float] = 0.02,
        lim_quantiles: Optional[tuple] = (0.002, 0.998),
        hold_end: Optional[float] = 1.0,
        save_to: Optional[str] = None,
        dpi: Optional[int] = 140,
    ) -> FuncAnimation:
    """
    Animate 2D Bass martingale paths in 3D: time vs. (coord 0, coord 1).

    Args:
        paths (np.ndarray): Array of shape (n_paths, n_steps, 2) representing the paths.
        mode (Literal["planar", "3d"]): Visualization mode: "planar" for 2D view (x=coord 0, y=coord 1), "3d" for 3D view (x=time, y=coord 0, z=coord 1).
        fps (Optional[int]): Frames per second for the animation.
        max_frames (Optional[int]): Maximum number of frames in the animation.
        frame_stride (Optional[str]): Frame selection strategy: "all", "auto", or an integer stride.
        figsize (Optional[tuple]): Figure size for the animation.
        view (Optional[tuple]): Elevation and azimuth angles for the 3D view.
        time_stretch (Optional[float]): Stretch factor for the time axis.
        point_size (Optional[float]): Size of the starting point markers.
        start_color (Optional[str]): Color for the starting points.
        thread_color (Optional[str]): Color for the path threads.
        end_color (Optional[str]): Color for the end points.
        thread_alpha (Optional[float]): Alpha transparency for the path threads.
        thread_width (Optional[float]): Line width for the path threads.
        max_threads (Optional[int]): Maximum number of threads to display.
        hide_axes (Optional[bool]): Whether to hide the axes.
        ylim (Optional[str]): Y-axis limits: "auto" or a tuple (ymin, ymax).
        zlim (Optional[str]): Z-axis limits: "auto" or a tuple (zmin, zmax).
        lim_pad (Optional[float]): Padding factor for automatic limits.
        lim_quantiles (Optional[tuple]): Quantiles for automatic limit calculation.
        hold_end (Optional[float]): Time in seconds to hold the last frame.
        save_to (Optional[str]): If provided, path to save the animation as an MP4 file.
        dpi (Optional[int]): Dots per inch for the saved animation.
    
    Returns:
        matplotlib.animation.FuncAnimation: The created animation object.
    """
    paths = np.asarray(paths)
    if paths.ndim != 3 or paths.shape[-1] != 2:
        raise ValueError("Expected shape (n_paths, n_steps, 2).")

    n_paths, n_steps, _ = paths.shape
    if n_steps < 2:
        raise ValueError("Need at least 2 steps.")
    if mode not in ("planar", "3d"):
        raise ValueError("mode must be 'planar' or '3d'.")

    knowing_hold = float(hold_end)
    if knowing_hold < 0.0:
        raise ValueError("hold_end must be >= 0.")

    def rgba(color, alpha=None):
        return to_rgba(color, alpha=alpha)

    def auto_lim(values, pad, q=None):
        v = np.asarray(values, float)
        lo = float(np.quantile(v, q[0])) if q is not None else float(np.min(v))
        hi = float(np.quantile(v, q[1])) if q is not None else float(np.max(v))
        span = hi - lo
        if span == 0.0:
            span = 1.0
        return lo - pad * span, hi + pad * span

    time_grid = np.linspace(0.0, 1.0, n_steps)

    if frame_stride == "all":
        frame_steps = np.arange(n_steps)
    elif frame_stride == "auto":
        n_frames = min(max_frames, n_steps)
        frame_steps = np.unique(np.round(np.linspace(0, n_steps - 1, n_frames)).astype(int))
    elif isinstance(frame_stride, int):
        if frame_stride <= 0:
            raise ValueError("frame_stride must be >= 1.")
        frame_steps = np.arange(0, n_steps, frame_stride)
    else:
        frame_steps = np.unique(np.asarray(frame_stride, dtype=int))
        frame_steps = frame_steps[(frame_steps >= 0) & (frame_steps < n_steps)]
        if len(frame_steps) == 0:
            raise ValueError("frame_stride produced no valid frames.")

    frame_steps = np.unique(np.r_[0, frame_steps, n_steps - 1])

    if knowing_hold > 0.0:
        extra = int(round(knowing_hold * fps))
        if extra > 0:
            frame_steps = np.r_[frame_steps, np.full(extra, frame_steps[-1], dtype=int)]

    y_all = paths[..., 0]
    z_all = paths[..., 1]
    y_limits = auto_lim(y_all, lim_pad, lim_quantiles) if ylim == "auto" else tuple(ylim)
    z_limits = auto_lim(z_all, lim_pad, lim_quantiles) if zlim == "auto" else tuple(zlim)

    if n_paths > max_threads:
        shown = np.linspace(0, n_paths - 1, max_threads).round().astype(int)
    else:
        shown = np.arange(n_paths)

    hair_color = rgba(thread_color, alpha=thread_alpha)
    c0 = np.array(rgba(start_color), dtype=float)
    c1 = np.array(rgba(end_color), dtype=float)

    if mode == "3d":
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.computed_zorder = False

        elev, azim = view
        ax.view_init(elev=elev, azim=azim)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(*y_limits)
        ax.set_zlim(*z_limits)

        x_span = 1.0
        y_span = (y_limits[1] - y_limits[0]) or 1.0
        z_span = (z_limits[1] - z_limits[0]) or 1.0
        ax.set_box_aspect((time_stretch * x_span, y_span, z_span))

        if hide_axes:
            ax.set_axis_off()
        else:
            ax.set_xlabel("t")
            ax.set_ylabel("coord 0")
            ax.set_zlabel("coord 1")

        time_text = ax.text2D(
            0.0, 1.0, "", transform=ax.transAxes, va="top", ha="left"
        )

        anchor_t = np.zeros(n_paths)
        anchor_y = paths[:, 0, 0]
        anchor_z = paths[:, 0, 1]

        ax.scatter(
            anchor_t, anchor_y, anchor_z,
            s=point_size, c=[rgba(start_color)],
            depthshade=False, zorder=1
        )

        threads = []
        for p in shown:
            (line,) = ax.plot(
                [0.0], [paths[p, 0, 0]], [paths[p, 0, 1]],
                lw=thread_width, color=hair_color, zorder=2
            )
            threads.append((line, p))

        moving = ax.scatter(
            anchor_t.copy(), anchor_y.copy(), anchor_z.copy(),
            s=point_size, c=[rgba(start_color)],
            depthshade=False, zorder=3
        )

        def update(i):
            step = int(frame_steps[i])
            t = float(time_grid[step])
            blend = i / (len(frame_steps) - 1) if len(frame_steps) > 1 else 1.0

            t_line = time_grid[: step + 1]
            for line, p in threads:
                line.set_data(t_line, paths[p, : step + 1, 0])
                line.set_3d_properties(paths[p, : step + 1, 1])

            moving._offsets3d = (
                np.full(n_paths, t),
                paths[:, step, 0],
                paths[:, step, 1],
            )

            color_now = (1.0 - blend) * c0 + blend * c1
            colors = np.tile(color_now, (n_paths, 1))
            moving.set_facecolors(colors)
            moving.set_edgecolors(colors)

            time_text.set_text(f"t = {t:.3f}")
            return (moving, time_text)

    else: # planar
        fig, ax = plt.subplots(figsize=figsize)

        ax.set_xlim(*y_limits)
        ax.set_ylim(*z_limits)
        ax.set_aspect("equal", adjustable="box")

        if hide_axes:
            ax.set_axis_off()
        else:
            ax.set_xlabel("coord 0")
            ax.set_ylabel("coord 1")

        time_text = ax.text(
            0.0, 1.0, "", transform=ax.transAxes, va="top", ha="left"
        )

        moving = ax.scatter(
            paths[:, 0, 0], paths[:, 0, 1],
            s=point_size, c=[rgba(start_color)],
            zorder=3
        )

        def update(i):
            step = int(frame_steps[i])
            t = float(time_grid[step])
            blend = i / (len(frame_steps) - 1) if len(frame_steps) > 1 else 1.0

            moving.set_offsets(np.c_[paths[:, step, 0], paths[:, step, 1]])

            color_now = (1.0 - blend) * c0 + blend * c1
            colors = np.tile(color_now, (n_paths, 1))
            moving.set_facecolors(colors)
            moving.set_edgecolors(colors)

            time_text.set_text(f"t = {t:.3f}")
            return (moving, time_text)

    anim = FuncAnimation(
        fig, update,
        frames=len(frame_steps),
        interval=int(round(1000 / fps)),
        blit=False,
    )

    if save_to is not None:
        anim.save(save_to, fps=fps, dpi=dpi)

    return anim
#endregion