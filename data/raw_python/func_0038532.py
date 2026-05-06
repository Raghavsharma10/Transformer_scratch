def generate_scaling_plot(timing_data, title, ylabel, description, plot_file):
    """
    Generate a scaling plot.

    Args:
        timing_data: data returned from a `*_scaling` method
        title: the title of the plot
        ylabel: the y-axis label of the plot
        description: a description of the plot
        plot_file: the file to write out to

    Returns:
        an image element containing the plot file and metadata
    """
    proc_counts = timing_data['proc_counts']
    if len(proc_counts) > 2:
        plt.figure(figsize=(10, 8), dpi=150)
        plt.title(title)
        plt.xlabel("Number of processors")
        plt.ylabel(ylabel)

        for case, case_color in zip(['bench', 'model'], ['#91bfdb', '#fc8d59']):
            case_data = timing_data[case]
            means = case_data['means']
            mins = case_data['mins']
            maxs = case_data['maxs']
            plt.fill_between(proc_counts, mins, maxs, facecolor=case_color, alpha=0.5)
            plt.plot(proc_counts, means, 'o-', color=case_color, label=case)

        plt.legend(loc='best')
    else:
        plt.figure(figsize=(5, 3))
        plt.axis('off')
        plt.text(0.4, 0.8, "ERROR:")
        plt.text(0.0, 0.6, "Not enough data points to draw scaling plot")
        plt.text(0.0, 0.44, "To generate this data rerun BATS with the")
        plt.text(0.0, 0.36, "performance option enabled.")

    if livvkit.publish:
        plt.savefig(os.path.splitext(plot_file)[0]+'.eps', dpi=600)
    plt.savefig(plot_file)
    plt.close()
    return elements.image(title, description, os.path.basename(plot_file))