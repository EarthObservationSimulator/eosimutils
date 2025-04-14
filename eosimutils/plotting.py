"""Module for plotting timeseries data."""

import matplotlib.pyplot as plt


def plot_timeseries(
    ts,
    cols: list = None,
    unused_colors: list = None,
    title: str = None,
    xlabel: str = "Time",
    ylabel: str = "Values",
    background: str = "white",
    grid: bool = True,
    **kwargs
):
    """
    Plots the timeseries data.

    Args:
        ts (Timeseries): The timeseries object to plot.
        cols (list, optional): List of column indices to plot. If None, all columns are plotted.
        unused_colors (list, optional): List of colors to use for the lines.
        title (str, optional): Title of the plot.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        background (str, optional): Background color for the plot.
        grid (bool, optional): Whether to show grid lines.
        **kwargs: Additional keyword arguments passed to the plot function.
    """
    fig, ax = plt.subplots()
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)

    lines = []
    col_offset = 0  # Track the column index across multiple arrays
    for arr, header in zip(ts.data, ts.headers):
        if isinstance(header, list):  # Vector data
            for i, subheader in enumerate(header):
                if cols is None or i + col_offset in cols:
                    (line,) = ax.plot(
                        ts.time.et, arr[:, i], label=subheader, **kwargs
                    )
                    lines.append(line)
            col_offset += len(header)
        else:  # Scalar data
            if cols is None or col_offset in cols:
                (line,) = ax.plot(ts.time.et, arr, label=header, **kwargs)
                lines.append(line)
            col_offset += 1

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if grid:
        ax.grid(True)
    ax.legend()

    return fig, ax, lines
