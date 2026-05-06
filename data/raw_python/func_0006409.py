def plot_profile_histogram(x, y, n_bins=100, title=None, x_label=None, y_label=None, log_y=False, filename=None):
    '''Takes 2D point data (x,y) and creates a profile histogram similar to the TProfile in ROOT. It calculates
    the y mean for every bin at the bin center and gives the y mean error as error bars.

    Parameters
    ----------
    x : array like
        data x positions
    y : array like
        data y positions
    n_bins : int
        the number of bins used to create the histogram
    '''
    if len(x) != len(y):
        raise ValueError('x and y dimensions have to be the same')
    n, bin_edges = np.histogram(x, bins=n_bins)  # needed to calculate the number of points per bin
    sy = np.histogram(x, bins=n_bins, weights=y)[0]  # the sum of the bin values
    sy2 = np.histogram(x, bins=n_bins, weights=y * y)[0]  # the quadratic sum of the bin values
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2  # calculate the bin center for all bins
    mean = sy / n  # calculate the mean of all bins
    std = np.sqrt((sy2 / n - mean * mean))  # TODO: not understood, need check if this is really the standard deviation
    #     std_mean = np.sqrt((sy2 - 2 * mean * sy + mean * mean) / (1*(n - 1)))  # this should be the formular ?!
    std_mean = std / np.sqrt((n - 1))
    mean[np.isnan(mean)] = 0.0
    std_mean[np.isnan(std_mean)] = 0.0

    fig = Figure()
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.errorbar(bin_centers, mean, yerr=std_mean, fmt='o')
    ax.set_title(title)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if log_y:
        ax.yscale('log')
    ax.grid(True)
    if not filename:
        fig.show()
    elif isinstance(filename, PdfPages):
        filename.savefig(fig)
    else:
        fig.savefig(filename)