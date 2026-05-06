def plot_linear_relation(x, y, x_err=None, y_err=None, title=None, point_label=None, legend=None, plot_range=None, plot_range_y=None, x_label=None, y_label=None, y_2_label=None, log_x=False, log_y=False, size=None, filename=None):
    ''' Takes point data (x,y) with errors(x,y) and fits a straight line. The deviation to this line is also plotted, showing the offset.

     Parameters
    ----------
    x, y, x_err, y_err: iterable

    filename: string, PdfPages object or None
        PdfPages file object: plot is appended to the pdf
        string: new plot file with the given filename is created
        None: the plot is printed to screen
    '''
    fig = Figure()
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    if x_err is not None:
        x_err = [x_err, x_err]
    if y_err is not None:
        y_err = [y_err, y_err]
    ax.set_title(title)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    if plot_range:
        ax.set_xlim((min(plot_range), max(plot_range)))
    if plot_range_y:
        ax.set_ylim((min(plot_range_y), max(plot_range_y)))
    if legend:
        fig.legend(legend, 0)
    ax.grid(True)
    ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='o', color='black')  # plot points
    # label points if needed
    if point_label is not None:
        for X, Y, Z in zip(x, y, point_label):
            ax.annotate('{}'.format(Z), xy=(X, Y), xytext=(-5, 5), ha='right', textcoords='offset points')
    line_fit, _ = np.polyfit(x, y, 1, full=False, cov=True)
    fit_fn = np.poly1d(line_fit)
    ax.plot(x, fit_fn(x), '-', lw=2, color='gray')
    setp(ax.get_xticklabels(), visible=False)  # remove ticks at common border of both plots

    divider = make_axes_locatable(ax)
    ax_bottom_plot = divider.append_axes("bottom", 2.0, pad=0.0, sharex=ax)

    ax_bottom_plot.bar(x, y - fit_fn(x), align='center', width=np.amin(np.diff(x)) / 2, color='gray')
#     plot(x, y - fit_fn(x))
    ax_bottom_plot.grid(True)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_2_label is not None:
        ax.set_ylabel(y_2_label)

    ax.set_ylim((-np.amax(np.abs(y - fit_fn(x)))), (np.amax(np.abs(y - fit_fn(x)))))

    ax.plot(ax.set_xlim(), [0, 0], '-', color='black')
    setp(ax_bottom_plot.get_yticklabels()[-2:-1], visible=False)

    if size is not None:
        fig.set_size_inches(size)

    if not filename:
        fig.show()
    elif isinstance(filename, PdfPages):
        filename.savefig(fig)
    elif filename:
        fig.savefig(filename, bbox_inches='tight')

    return fig