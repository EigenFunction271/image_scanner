"""Visualization functions for optics consistency analysis."""

import logging
import textwrap
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from image_screener.optics_consistency import OpticsConsistencyResult

logger = logging.getLogger(__name__)


def create_optics_diagnostics_plot(
    result: OpticsConsistencyResult,
    image_path: str,
    output_path: Optional[Path] = None,
    show_plot: bool = True,
) -> None:
    """
    Create comprehensive diagnostic plots for optics consistency analysis.

    Args:
        result: OpticsConsistencyResult from analysis
        image_path: Path to original image
        output_path: Optional path to save plot
        show_plot: Whether to display plot interactively
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.5)

    # Title
    filename = Path(image_path).name
    fig.suptitle(
        f'Analysis of "{filename}": Overall Score {result.optics_score:.3f}',
        fontsize=16,
        fontweight="bold",
    )

    # 1. Radial Power Spectrum (Frequency Test)
    ax1 = fig.add_subplot(gs[0, 0])
    freq_data = result.frequency_test.diagnostic_data
    if "radii" in freq_data and "radial_power" in freq_data:
        radii = freq_data["radii"]
        radial_power = freq_data["radial_power"]

        ax1.plot(radii, radial_power, "b-", linewidth=2, label="Radial Power")
        if "predicted" in freq_data and "log_radii" in freq_data:
            # Plot fitted line in log-log space
            log_radii = freq_data["log_radii"]
            predicted = freq_data["predicted"]
            # Convert log_radii back to normalized radii for plotting
            # log_radii = log(radii), so radii = exp(log_radii)
            # But we need to map back to the original radii scale
            # Find the corresponding radii values
            valid_mask = (radii > 0.1) & (radii < 0.9)
            if np.sum(valid_mask) > 0:
                plot_radii = radii[valid_mask]
                ax1.plot(plot_radii, predicted, "r--", linewidth=1.5, label="Fitted Decay")

        ax1.set_xlabel("Normalized Radius", fontweight="bold")
        ax1.set_ylabel("Log Power", fontweight="bold")
        ax1.set_title(
            f"Frequency Test\nScore: {result.frequency_test.score:.3f}",
            fontweight="bold",
        )
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Add violations text and new metrics
        violations_text = "\n".join(result.frequency_test.violations[:2])
        
        # Add new metrics information
        metrics_lines = []
        if "high_freq_energy" in freq_data:
            metrics_lines.append(f"HF Energy: {freq_data['high_freq_energy']:.4f}")
        if "mid_bump_ratio" in freq_data:
            metrics_lines.append(f"Mid Bump: {freq_data['mid_bump_ratio']:.2f}x")
        if "spatial_variance" in freq_data:
            metrics_lines.append(f"Spatial Var: {freq_data['spatial_variance']:.4f}")
        
        if metrics_lines:
            metrics_text = " | ".join(metrics_lines)
            violations_text = violations_text + "\n" + metrics_text if violations_text else metrics_text
        
        ax1.text(
            0.02,
            0.98,
            violations_text,
            transform=ax1.transAxes,
            fontsize=7,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # 2. ESF/LSF Example (Edge PSF Test)
    ax2 = fig.add_subplot(gs[0, 1])
    psf_data = result.edge_psf_test.diagnostic_data
    if "example_esf" in psf_data and psf_data["example_esf"] is not None:
        esf = psf_data["example_esf"]
        lsf = psf_data["example_lsf"]

        x_esf = np.arange(len(esf)) - len(esf) // 2
        x_lsf = np.arange(len(lsf)) - len(lsf) // 2

        ax2_twin = ax2.twinx()
        line1 = ax2.plot(x_esf, esf, "b-", linewidth=2, label="ESF", alpha=0.7)
        line2 = ax2_twin.plot(
            x_lsf, lsf, "r-", linewidth=2, label="LSF", alpha=0.7
        )

        ax2.set_xlabel("Distance from Edge (pixels)", fontweight="bold")
        ax2.set_ylabel("ESF Intensity", fontweight="bold", color="b")
        ax2_twin.set_ylabel("LSF Magnitude", fontweight="bold", color="r")
        ax2.set_title(
            f"Edge PSF Test\nScore: {result.edge_psf_test.score:.3f}",
            fontweight="bold",
        )
        ax2.grid(True, alpha=0.3)

        # Combine legends - place at bottom right to avoid overlap with violation text
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc="lower right")

        # Add violations text
        violations_text = "\n".join(result.edge_psf_test.violations[:2])
        ax2.text(
            0.02,
            0.98,
            violations_text,
            transform=ax2.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
    else:
        ax2.text(0.5, 0.5, "No ESF data available", ha="center", va="center")
        ax2.set_title(
            f"Edge PSF Test\nScore: {result.edge_psf_test.score:.3f}",
            fontweight="bold",
        )

    # 3. Blur Radius Heatmap (DOF Test)
    ax3 = fig.add_subplot(gs[0, 2])
    dof_data = result.dof_consistency_test.diagnostic_data
    if "blur_map" in dof_data:
        blur_map = dof_data["blur_map"]
        # Handle NaN values: mask them for visualization
        # This prevents matplotlib from normalizing incorrectly when most values are NaN
        blur_map_vis = np.ma.masked_invalid(blur_map)
        im = ax3.imshow(blur_map_vis, cmap="hot", aspect="auto", interpolation="bilinear")
        ax3.set_title(
            f"DOF Consistency Test\nScore: {result.dof_consistency_test.score:.3f}",
            fontweight="bold",
        )
        ax3.set_xlabel("X (sampled)", fontweight="bold")
        ax3.set_ylabel("Y (sampled)", fontweight="bold")
        plt.colorbar(im, ax=ax3, label="Blur Radius")

        # Add violations text - wrap long text to prevent overflow
        violations_text = "\n".join(result.dof_consistency_test.violations[:2])
        # Wrap text at 70 characters per line to prevent overflow
        wrapped_lines = []
        for line in violations_text.split("\n"):
            wrapped_lines.extend(textwrap.wrap(line, width=70, break_long_words=False))
        violations_text = "\n".join(wrapped_lines[:3])  # Limit to 3 lines max
        
        ax3.text(
            0.02,
            0.98,
            violations_text,
            transform=ax3.transAxes,
            fontsize=7,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
    else:
        ax3.text(0.5, 0.5, "No blur map data available", ha="center", va="center")
        ax3.set_title(
            f"DOF Consistency Test\nScore: {result.dof_consistency_test.score:.3f}",
            fontweight="bold",
        )

    # 4. Chromatic Aberration Offsets
    ax4 = fig.add_subplot(gs[1, 0])
    ca_data = result.chromatic_aberration_test.diagnostic_data
    if "rg_offsets" in ca_data and "bg_offsets" in ca_data:
        rg_offsets = ca_data["rg_offsets"]
        bg_offsets = ca_data["bg_offsets"]

        ax4.hist(
            rg_offsets,
            bins=20,
            alpha=0.6,
            label=f"R-G (mean: {np.mean(rg_offsets):.2f})",
            color="red",
        )
        ax4.hist(
            bg_offsets,
            bins=20,
            alpha=0.6,
            label=f"B-G (mean: {np.mean(bg_offsets):.2f})",
            color="blue",
        )
        ax4.set_xlabel("Edge Offset (pixels)", fontweight="bold")
        ax4.set_ylabel("Frequency", fontweight="bold")
        ax4.set_title(
            f"Chromatic Aberration Test\nScore: {result.chromatic_aberration_test.score:.3f}",
            fontweight="bold",
        )
        # Position legend at bottom-right to avoid overlap with violation text
        ax4.legend(loc="lower right")
        ax4.grid(True, alpha=0.3)

        # Add violations text - wrap long text to prevent overflow
        violations_text = "\n".join(result.chromatic_aberration_test.violations[:2])
        # Wrap text at 70 characters per line to prevent overflow
        wrapped_lines = []
        for line in violations_text.split("\n"):
            wrapped_lines.extend(textwrap.wrap(line, width=70, break_long_words=False))
        violations_text = "\n".join(wrapped_lines[:3])  # Limit to 3 lines max
        
        ax4.text(
            0.02,
            0.98,
            violations_text,
            transform=ax4.transAxes,
            fontsize=7,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
    else:
        ax4.text(0.5, 0.5, "No CA data available", ha="center", va="center")
        ax4.set_title(
            f"Chromatic Aberration Test\nScore: {result.chromatic_aberration_test.score:.3f}",
            fontweight="bold",
        )

    # 5. Original Image
    ax5 = fig.add_subplot(gs[1, 1])
    try:
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_array = np.array(img)
        ax5.imshow(img_array)
        ax5.set_title("Original Image", fontweight="bold")
        ax5.axis("off")
    except Exception as e:
        logger.warning(f"Failed to load original image: {e}")
        ax5.text(0.5, 0.5, "Failed to load image", ha="center", va="center")
        ax5.set_title("Original Image", fontweight="bold")

    # 6. Noise Residual Visualization (moved from row 2 to replace Test Scores Summary)
    ax6 = fig.add_subplot(gs[1, 2])
    noise_data = result.noise_residual_test.diagnostic_data
    if "noise_residual" in noise_data and noise_data["noise_residual"] is not None:
        noise_residual = noise_data["noise_residual"]
        # Downsample if too large for visualization
        if noise_residual.shape[0] > 256 or noise_residual.shape[1] > 256:
            from scipy.ndimage import zoom
            zoom_factor = min(256 / noise_residual.shape[0], 256 / noise_residual.shape[1])
            noise_residual = zoom(noise_residual, zoom_factor)
        
        im = ax6.imshow(
            noise_residual,
            cmap="gray",
            aspect="auto",
            interpolation="bilinear",
        )
        ax6.set_title(
            f"Noise Residual Test\nScore: {result.noise_residual_test.score:.3f}",
            fontweight="bold",
        )
        ax6.set_xlabel("X (pixels)", fontweight="bold")
        ax6.set_ylabel("Y (pixels)", fontweight="bold")
        plt.colorbar(im, ax=ax6, label="Residual")
        
        # Add violations text
        violations_text = "\n".join(result.noise_residual_test.violations[:2])
        wrapped_lines = []
        for line in violations_text.split("\n"):
            wrapped_lines.extend(textwrap.wrap(line, width=70, break_long_words=False))
        violations_text = "\n".join(wrapped_lines[:3])
        
        ax6.text(
            0.02,
            0.98,
            violations_text,
            transform=ax6.transAxes,
            fontsize=7,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
    else:
        ax6.text(0.5, 0.5, "No noise residual data", ha="center", va="center")
        ax6.set_title(
            f"Noise Residual Test\nScore: {result.noise_residual_test.score:.3f}",
            fontweight="bold",
        )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved diagnostics plot to {output_path}")

    if show_plot:
        # Only show if backend supports it
        backend = plt.get_backend()
        if backend.lower() != "agg":
            plt.show()
        else:
            logger.debug(f"Skipping plt.show() - non-interactive backend: {backend}")
    else:
        plt.close(fig)

