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
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.5)

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

        # Add violations text
        violations_text = "\n".join(result.frequency_test.violations[:2])
        ax1.text(
            0.02,
            0.98,
            violations_text,
            transform=ax1.transAxes,
            fontsize=8,
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
        im = ax3.imshow(blur_map, cmap="hot", aspect="auto", interpolation="bilinear")
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

    # 5. Noise Residual Autocorrelation (Noise Residual Test)
    ax5 = fig.add_subplot(gs[1, 0])
    noise_data = result.noise_residual_test.diagnostic_data
    if "autocorrelation_2d" in noise_data and noise_data["autocorrelation_2d"] is not None:
        autocorr_2d = noise_data["autocorrelation_2d"]
        # Extract center region for visualization (e.g., 50x50 around center)
        center_y, center_x = autocorr_2d.shape[0] // 2, autocorr_2d.shape[1] // 2
        region_size = min(50, autocorr_2d.shape[0] // 4, autocorr_2d.shape[1] // 4)
        autocorr_vis = autocorr_2d[
            center_y - region_size : center_y + region_size + 1,
            center_x - region_size : center_x + region_size + 1,
        ]
        
        im = ax5.imshow(
            autocorr_vis,
            cmap="coolwarm",
            aspect="auto",
            interpolation="bilinear",
            vmin=-1.0,
            vmax=1.0,
        )
        ax5.set_title(
            f"Noise Residual Test\nScore: {result.noise_residual_test.score:.3f}",
            fontweight="bold",
        )
        ax5.set_xlabel("X (offset)", fontweight="bold")
        ax5.set_ylabel("Y (offset)", fontweight="bold")
        plt.colorbar(im, ax=ax5, label="Correlation")
        
        # Mark center point
        ax5.plot(region_size, region_size, "ko", markersize=5, label="Center")
        
        # Add violations text
        violations_text = "\n".join(result.noise_residual_test.violations[:2])
        wrapped_lines = []
        for line in violations_text.split("\n"):
            wrapped_lines.extend(textwrap.wrap(line, width=70, break_long_words=False))
        violations_text = "\n".join(wrapped_lines[:3])
        
        ax5.text(
            0.02,
            0.98,
            violations_text,
            transform=ax5.transAxes,
            fontsize=7,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
    else:
        ax5.text(0.5, 0.5, "No autocorrelation data available", ha="center", va="center")
        ax5.set_title(
            f"Noise Residual Test\nScore: {result.noise_residual_test.score:.3f}",
            fontweight="bold",
        )

    # 6. Original Image
    ax6 = fig.add_subplot(gs[1, 1])
    try:
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_array = np.array(img)
        ax6.imshow(img_array)
        ax6.set_title("Original Image", fontweight="bold")
        ax6.axis("off")
    except Exception as e:
        logger.warning(f"Failed to load original image: {e}")
        ax6.text(0.5, 0.5, "Failed to load image", ha="center", va="center")
        ax6.set_title("Original Image", fontweight="bold")

    # 7. Score Summary
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.axis("off")

    scores = [
        ("Frequency Test", result.frequency_test.score),
        ("Edge PSF Test", result.edge_psf_test.score),
        ("DOF Consistency", result.dof_consistency_test.score),
        ("Chromatic Aberration", result.chromatic_aberration_test.score),
        ("Noise Residual", result.noise_residual_test.score),
        ("Overall Score", result.optics_score),
    ]

    y_pos = np.arange(len(scores))
    colors = [
        "green" if s >= 0.7 else "orange" if s >= 0.4 else "red"
        for _, s in scores
    ]

    bars = ax7.barh(y_pos, [s for _, s in scores], color=colors, alpha=0.7)
    ax7.set_yticks(y_pos)
    ax7.set_yticklabels([name for name, _ in scores])
    ax7.set_xlabel("Score (0.0 - 1.0)", fontweight="bold")
    ax7.set_title("Test Scores Summary", fontweight="bold")
    ax7.set_xlim(0, 1.0)
    ax7.grid(True, alpha=0.3, axis="x")

    # Add value labels on bars
    for i, (bar, (_, score)) in enumerate(zip(bars, scores)):
        ax7.text(
            score + 0.02,
            i,
            f"{score:.3f}",
            va="center",
            fontweight="bold",
        )
    
    # 8. Noise Residual Visualization
    ax8 = fig.add_subplot(gs[2, 0])
    if "noise_residual" in noise_data and noise_data["noise_residual"] is not None:
        noise_residual = noise_data["noise_residual"]
        # Downsample if too large for visualization
        if noise_residual.shape[0] > 256 or noise_residual.shape[1] > 256:
            from scipy.ndimage import zoom
            zoom_factor = min(256 / noise_residual.shape[0], 256 / noise_residual.shape[1])
            noise_residual = zoom(noise_residual, zoom_factor)
        
        im = ax8.imshow(
            noise_residual,
            cmap="gray",
            aspect="auto",
            interpolation="bilinear",
        )
        ax8.set_title("Noise Residual", fontweight="bold")
        ax8.set_xlabel("X (pixels)", fontweight="bold")
        ax8.set_ylabel("Y (pixels)", fontweight="bold")
        plt.colorbar(im, ax=ax8, label="Residual")
    else:
        ax8.text(0.5, 0.5, "No noise residual data", ha="center", va="center")
        ax8.set_title("Noise Residual", fontweight="bold")
    
    # 9. Correlation Statistics
    ax9 = fig.add_subplot(gs[2, 1])
    ax9.axis("off")
    if "mean_correlation" in noise_data and "decorrelation_factor" in noise_data:
        mean_corr = noise_data["mean_correlation"]
        decorr_factor = noise_data["decorrelation_factor"]
        
        stats_text = f"Mean Correlation (ρ): {mean_corr:.4f}\n"
        stats_text += f"Decorrelation Factor: {decorr_factor:.4f}\n"
        stats_text += f"Threshold: {0.15:.2f}\n\n"
        
        if mean_corr < 0.15:
            stats_text += "Status: DECORRELATED\n(AI-generated)"
        else:
            stats_text += "Status: CORRELATED\n(Real sensor)"
        
        ax9.text(
            0.5,
            0.5,
            stats_text,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        )
        ax9.set_title("Noise Correlation Statistics", fontweight="bold")
    
    # 10. Empty subplot (can be used for future visualizations)
    ax10 = fig.add_subplot(gs[2, 2])
    ax10.axis("off")

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

