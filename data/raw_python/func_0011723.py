def _plot_variability(ts, variability, threshold=None, epochs=None):
    """Plot the timeseries and variability. Optionally plot epochs."""
    import matplotlib.style
    import matplotlib as mpl
    mpl.style.use('classic')
    import matplotlib.pyplot as plt
    if variability.ndim is 1:
        variability = variability[:, np.newaxis, np.newaxis]
    elif variability.ndim is 2:
        variability = variability[:, np.newaxis, :]
    vmeasures = variability.shape[2]
    channels = ts.shape[1]
    dt = (1.0*ts.tspan[-1] - ts.tspan[0]) / (len(ts) - 1)
    fig = plt.figure()
    ylabelprops = dict(rotation=0, 
                       horizontalalignment='right', 
                       verticalalignment='center', 
                       x=-0.01)
    for i in range(channels):
        rect = (0.1, 0.85*(channels - i - 1)/channels + 0.1, 
                0.8, 0.85/channels)
        axprops = dict()
        if channels > 10:
            axprops['yticks'] = []
        ax = fig.add_axes(rect, **axprops)
        ax.plot(ts.tspan, ts[:, i])
        if ts.labels[1] is None:
            ax.set_ylabel(u'channel %d' % i, **ylabelprops)
        else:
            ax.set_ylabel(ts.labels[1][i], **ylabelprops)
        plt.setp(ax.get_xticklabels(), visible=False)
        if i is channels - 1:
            plt.setp(ax.get_xticklabels(), visible=True)
            ax.set_xlabel('time (s)')
        ax2 = ax.twinx()
        if vmeasures > 1:
            mean_v = np.nanmean(variability[:, i, :], axis=1)
            ax2.plot(ts.tspan, mean_v, color='g')
            colors = _get_color_list()
            for j in range(vmeasures):
                ax2.plot(ts.tspan, variability[:, i, j], linestyle='dotted',
                         color=colors[(3 + j) % len(colors)])
            if i is 0:
                ax2.legend(['variability (mean)'] + 
                          ['variability %d' % j for j in range(vmeasures)], 
                          loc='best')
        else:
            ax2.plot(ts.tspan, variability[:, i, 0])
            ax2.legend(('variability',), loc='best')
        if threshold is not None:
            ax2.axhline(y=threshold, color='Gray', linestyle='dashed')
        ax2.set_ylabel('variability')
        ymin = np.nanmin(ts[:, i])
        ymax = np.nanmax(ts[:, i])
        tstart = ts.tspan[0]
        if epochs:
            # highlight epochs using rectangular patches
            for e in epochs[i]:
                t1 = tstart + (e[0] - 1) * dt
                ax.add_patch(mpl.patches.Rectangle(
                    (t1, ymin), (e[1] - e[0])*dt, ymax - ymin, alpha=0.2,
                    color='green', ec='none'))
    fig.axes[0].set_title(u'variability (threshold = %g)' % threshold)
    fig.show()