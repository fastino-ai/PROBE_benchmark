"""
Visualization of context size impact on model performance across frontier models.
Data extracted from performance analysis chart.

ICLR conference paper styling:
- Figure width: 5.5" (single column) or 6.75" (1.5 column)
- Font: Serif family (Times/Palatino), 8-10pt
- Grayscale-friendly with patterns
- High resolution (300 DPI)

Usage:
    # Generate all figures (saves to figures/ directory)
    python visualisation.py

    # Display figures interactively
    python visualisation.py --show

    # Print summary only
    python visualisation.py --summary

    # Custom output directory
    python visualisation.py --output-dir my_figures/

Import as module:
    from visualisation import plot_performance_degradation, PERFORMANCE_DATA

    fig = plot_performance_degradation(figure_width=5.5)
    # Access data: PERFORMANCE_DATA["GPT-5"][100] -> (0.377, -15.5)
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import Dict, Tuple


# Configure matplotlib for publication quality
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
mpl.rcParams["font.size"] = 9
mpl.rcParams["axes.labelsize"] = 10
mpl.rcParams["axes.titlesize"] = 11
mpl.rcParams["xtick.labelsize"] = 9
mpl.rcParams["ytick.labelsize"] = 9
mpl.rcParams["legend.fontsize"] = 9
mpl.rcParams["figure.titlesize"] = 11
mpl.rcParams["pdf.fonttype"] = 42  # TrueType fonts for editability
mpl.rcParams["ps.fonttype"] = 42

# Performance data: {model: {num_distractors: (score, percentage_change)}}
PERFORMANCE_DATA: Dict[str, Dict[int, Tuple[float, float]]] = {
    "Claude Sonnet 4": {
        50: (0.448, 0.0),  # Baseline
        75: (0.407, -9.2),
        100: (0.256, -37.0),
    },
    "Claude Opus 4.1": {
        50: (0.468, 0.0),  # Baseline
        75: (0.435, -7.1),
        100: (0.293, -32.7),
    },
    "GPT-5": {
        50: (0.484, 0.0),  # Baseline
        75: (0.447, -7.8),
        100: (0.377, -15.5),
    },
}


def plot_performance_degradation(
    figure_width: float = 6.75, save_path: str = None
) -> plt.Figure:
    """
    Create ICLR-style bar chart showing performance degradation across models.

    Args:
        figure_width: Width in inches (5.5 for single column, 6.75 for 1.5 column)
        save_path: Optional path to save figure (PDF recommended for papers)

    Returns:
        Matplotlib figure object
    """
    models = list(PERFORMANCE_DATA.keys())
    distractor_levels = [50, 75, 100]
    n_levels = len(distractor_levels)
    n_models = len(models)

    # Elegant, sophisticated color palette for academic publications
    # Muted tones that are refined yet distinctive and colorblind-friendly
    colors = ["#C85A54", "#5B8FAE", "#7FA879"]  # Muted coral, slate blue, sage green

    # Set up figure with ICLR dimensions
    fig_height = figure_width * 0.65  # Maintain good aspect ratio
    fig, ax = plt.subplots(figsize=(figure_width, fig_height))

    # Bar configuration
    bar_width = 0.25
    x = np.arange(n_levels)

    # Plot bars for each model
    for i, model in enumerate(models):
        offset = (i - n_models / 2 + 0.5) * bar_width
        scores = [PERFORMANCE_DATA[model][d][0] for d in distractor_levels]

        bars = ax.bar(
            x + offset,
            scores,
            bar_width,
            label=model,
            color=colors[i],
            edgecolor="white",
            linewidth=1.5,
            alpha=0.9,
        )

        # Add value annotations on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="semibold",
            )

    # Enhanced styling with emphasis on context window size
    ax.set_ylabel("Performance Score", fontweight="semibold")
    ax.set_xlabel(
        "Context Window Size (Number of Distractor Items)", fontweight="semibold"
    )
    ax.set_title(
        "Performance Degradation with Increasing Context Window Size",
        fontweight="bold",
        pad=15,
    )

    # X-axis configuration with baseline indication
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{d}\n(+{d-50})" if d > 50 else f"{d}\n(baseline)" for d in distractor_levels]
    )

    # Y-axis configuration
    ax.set_ylim(0, 0.55)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

    # Enhanced grid for readability
    ax.grid(
        axis="y", alpha=0.2, linestyle="-", linewidth=0.8, which="major", color="gray"
    )
    ax.grid(
        axis="y", alpha=0.1, linestyle="-", linewidth=0.5, which="minor", color="gray"
    )
    ax.set_axisbelow(True)

    # Better legend with styling
    legend = ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.98,
        edgecolor="#333333",
        title="Model",
        title_fontsize=10,
        fontsize=9,
    )
    legend.get_frame().set_linewidth(1.2)

    # Clean spines with better styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            format="pdf" if save_path.endswith(".pdf") else None,
        )
        print(f"Figure saved to: {save_path}")

    return fig


def get_performance_summary() -> str:
    """
    Generate a text summary of performance degradation.

    Returns:
        Formatted string with performance summary
    """
    summary_lines = ["Performance Degradation Summary", "=" * 50, ""]

    for model, data in PERFORMANCE_DATA.items():
        summary_lines.append(f"{model}:")
        for distractors in [50, 75, 100]:
            score, pct_change = data[distractors]
            summary_lines.append(
                f"  {distractors} distractors: {score:.3f} ({pct_change:+.1f}%)"
            )
        summary_lines.append("")

    return "\n".join(summary_lines)


def plot_degradation_lines(
    figure_width: float = 5.5, save_path: str = None
) -> plt.Figure:
    """
    Create ICLR-style line plot showing performance degradation trends.
    Alternative visualization to the bar chart.

    Args:
        figure_width: Width in inches (5.5 for single column)
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure object
    """
    models = list(PERFORMANCE_DATA.keys())
    distractor_levels = [50, 75, 100]

    # Elegant, sophisticated color palette (matching bar plot)
    # Muted tones that are refined yet distinctive and colorblind-friendly
    colors = ["#C85A54", "#5B8FAE", "#7FA879"]  # Muted coral, slate blue, sage green
    markers = ["o", "s", "D"]  # Circle, Square, Diamond
    linestyles = ["-", "-", "-"]  # Solid lines for all

    fig_height = figure_width * 0.7
    fig, ax = plt.subplots(figsize=(figure_width, fig_height))

    # Plot lines with enhanced styling
    for i, model in enumerate(models):
        scores = [PERFORMANCE_DATA[model][d][0] for d in distractor_levels]

        # Main line
        ax.plot(
            distractor_levels,
            scores,
            marker=markers[i],
            linestyle=linestyles[i],
            linewidth=2.5,
            markersize=9,
            color=colors[i],
            label=model,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=3,
            alpha=0.9,
        )

        # Add subtle shadow for depth
        ax.plot(
            distractor_levels,
            scores,
            marker=markers[i],
            linestyle=linestyles[i],
            linewidth=3,
            markersize=10,
            color="black",
            alpha=0.1,
            zorder=2,
        )

    # Enhanced labels emphasizing context window size
    ax.set_xlabel(
        "Context Window Size (Number of Distractor Items)", fontweight="semibold"
    )
    ax.set_ylabel("Performance Score", fontweight="semibold")
    ax.set_title(
        "Performance Degradation with Increasing Context Window Size",
        fontweight="bold",
        pad=15,
    )

    # X-axis configuration
    ax.set_xticks(distractor_levels)
    ax.set_xticklabels(
        [
            f"{d}\n(+{d-50} items)" if d > 50 else f"{d}\n(baseline)"
            for d in distractor_levels
        ]
    )

    # Y-axis configuration with better range
    ax.set_ylim(0.2, 0.52)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.025))

    # Enhanced grid
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.8, which="major", color="gray")
    ax.grid(True, alpha=0.1, linestyle="-", linewidth=0.5, which="minor", color="gray")
    ax.set_axisbelow(True)

    # Better legend with border
    legend = ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.98,
        edgecolor="#333333",
        fancybox=True,
        shadow=True,
        fontsize=9,
        title="Model",
        title_fontsize=10,
    )
    legend.get_frame().set_linewidth(1.2)

    # Clean spines with better styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            format="pdf" if save_path.endswith(".pdf") else None,
        )
        print(f"Figure saved to: {save_path}")

    return fig


def plot_relative_degradation(
    figure_width: float = 5.5, save_path: str = None
) -> plt.Figure:
    """
    Create ICLR-style bar chart showing relative performance degradation.
    Shows percentage change from baseline.

    Args:
        figure_width: Width in inches
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure object
    """
    models = list(PERFORMANCE_DATA.keys())
    distractor_levels = [75, 100]  # Exclude baseline
    n_levels = len(distractor_levels)
    n_models = len(models)

    # Elegant, sophisticated color palette (matching other plots)
    colors = ["#C85A54", "#5B8FAE", "#7FA879"]  # Muted coral, slate blue, sage green

    fig_height = figure_width * 0.65
    fig, ax = plt.subplots(figsize=(figure_width, fig_height))

    bar_width = 0.25
    x = np.arange(n_levels)

    for i, model in enumerate(models):
        offset = (i - n_models / 2 + 0.5) * bar_width
        percentages = [PERFORMANCE_DATA[model][d][1] for d in distractor_levels]

        bars = ax.bar(
            x + offset,
            percentages,
            bar_width,
            label=model,
            color=colors[i],
            edgecolor="white",
            linewidth=1.5,
            alpha=0.9,
        )

        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height - 2 if height < -10 else height + 1,
                f"{pct:.1f}%",
                ha="center",
                va="top" if height < -10 else "bottom",
                fontsize=8,
                fontweight="semibold",
            )

    # Enhanced labels with emphasis on context window
    ax.set_ylabel("Performance Change (%)", fontweight="semibold")
    ax.set_xlabel(
        "Context Window Size (Number of Distractor Items)", fontweight="semibold"
    )
    ax.set_title(
        "Relative Performance Degradation from Baseline (50 Distractors)",
        fontweight="bold",
        pad=15,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n(+{d-50})" for d in distractor_levels])
    ax.axhline(y=0, color="#333333", linewidth=1.2, linestyle="-", alpha=0.7)

    # Enhanced grid
    ax.grid(axis="y", alpha=0.2, linestyle="-", linewidth=0.8, color="gray")
    ax.set_axisbelow(True)

    # Better legend
    legend = ax.legend(
        loc="lower left",
        frameon=True,
        framealpha=0.98,
        edgecolor="#333333",
        fancybox=True,
        shadow=True,
        title="Model",
        title_fontsize=10,
        fontsize=9,
    )
    legend.get_frame().set_linewidth(1.2)

    # Clean spines with better styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            format="pdf" if save_path.endswith(".pdf") else None,
        )
        print(f"Figure saved to: {save_path}")

    return fig


def analyze_degradation_rates() -> Dict[str, float]:
    """
    Calculate the rate of performance degradation per distractor added.

    Returns:
        Dictionary mapping model names to degradation rates
    """
    rates = {}

    for model, data in PERFORMANCE_DATA.items():
        baseline = data[50][0]
        final = data[100][0]
        total_degradation = final - baseline
        num_distractors_added = 50  # From 50 to 100
        rate = total_degradation / num_distractors_added
        rates[model] = rate

    return rates


def get_latex_figure_code(
    figure_name: str = "performance_degradation_bars",
    width: str = "\\textwidth",
    label: str = "fig:performance_degradation",
    caption: str = None,
) -> str:
    """
    Generate LaTeX code for including figure in ICLR paper.

    Args:
        figure_name: Name of figure file (without extension)
        width: Width specification (e.g., "\\textwidth", "0.8\\textwidth")
        label: LaTeX label for referencing
        caption: Figure caption (auto-generated if None)

    Returns:
        LaTeX code string
    """
    if caption is None:
        caption = (
            "Performance degradation across frontier models with increasing "
            "context size. Models show varying degrees of performance loss as "
            "the number of distractor items increases from 50 to 100. GPT-5 "
            "demonstrates the most robust performance, maintaining 77.9\\% of "
            "baseline performance at 100 distractors, while Claude Sonnet 4 "
            "retains only 57.1\\%."
        )

    latex_code = f"""\\begin{{figure}}[t]
    \\centering
    \\includegraphics[width={width}]{{figures/{figure_name}.pdf}}
    \\caption{{{caption}}}
    \\label{{{label}}}
\\end{{figure}}"""

    return latex_code


def generate_all_figures(output_dir: str = "figures") -> None:
    """
    Generate all figure variations for ICLR paper.

    Args:
        output_dir: Directory to save figures (created if doesn't exist)
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    print("Generating ICLR-style figures...")
    print("=" * 60)

    # Figure 1: Main bar chart (1.5 column width)
    print("\n1. Main performance degradation chart (bar plot)...")
    plot_performance_degradation(
        figure_width=6.75, save_path=f"{output_dir}/performance_degradation_bars.pdf"
    )

    # Figure 2: Line plot alternative (single column)
    print("2. Performance degradation trends (line plot)...")
    plot_degradation_lines(
        figure_width=5.5, save_path=f"{output_dir}/performance_degradation_lines.pdf"
    )

    # Figure 3: Relative degradation (single column)
    print("3. Relative performance degradation...")
    plot_relative_degradation(
        figure_width=5.5, save_path=f"{output_dir}/relative_degradation.pdf"
    )

    print("\n" + "=" * 60)
    print(f"All figures saved to '{output_dir}/' directory")
    print("\nRecommendations for ICLR paper:")
    print("  - Use PDF format for vector graphics (included)")
    print("  - Main figure: performance_degradation_bars.pdf (1.5 column)")
    print("  - Alternative: performance_degradation_lines.pdf (single column)")
    print("  - Supplement: relative_degradation.pdf")

    print("\n" + "=" * 60)
    print("LaTeX code for main figure:")
    print("=" * 60)
    print(get_latex_figure_code())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate ICLR-style visualizations for performance analysis"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Directory to save figures (default: figures/)",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display figures interactively"
    )
    parser.add_argument(
        "--summary", action="store_true", help="Print performance summary only"
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Print LaTeX code for including figure in paper",
    )

    args = parser.parse_args()

    # Print summary
    print(get_performance_summary())

    # Print degradation rates
    print("\nDegradation Rate per Additional Distractor:")
    print("=" * 60)
    rates = analyze_degradation_rates()
    for model, rate in sorted(rates.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"{model:20s}: {rate:.6f} points/distractor ({rate*100:.4f}%)")

    # Print LaTeX code if requested
    if args.latex:
        print("\n" + "=" * 60)
        print("LaTeX code for main figure:")
        print("=" * 60)
        print(get_latex_figure_code())
        print("\n" + "=" * 60)
        print("LaTeX code for line plot:")
        print("=" * 60)
        print(
            get_latex_figure_code(
                figure_name="performance_degradation_lines",
                label="fig:degradation_lines",
                width="0.8\\textwidth",
            )
        )

    if not args.summary:
        # Generate all figures
        generate_all_figures(args.output_dir)

        # Show interactively if requested
        if args.show:
            print("\nDisplaying figures...")
            plot_performance_degradation()
            plot_degradation_lines()
            plot_relative_degradation()
            plt.show()
