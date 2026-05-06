def add_significance_indicator(plot, col_a=0, col_b=1, significant=False):
    """
    Add a p-value significance indicator.
    """
    plot_bottom, plot_top = plot.get_ylim()
    # Give the plot a little room for the significance indicator
    line_height = vertical_percent(plot, 0.1)
    # Add some extra spacing below the indicator
    plot_top = plot_top + line_height
    # Add some extra spacing above the indicator
    plot.set_ylim(top=plot_top + line_height * 2)
    color = "black"
    line_top = plot_top + line_height
    plot.plot([col_a, col_a, col_b, col_b], [plot_top, line_top, line_top, plot_top], lw=1.5, color=color)
    indicator = "*" if significant else "ns"
    plot.text((col_a + col_b) * 0.5, line_top, indicator, ha="center", va="bottom", color=color)