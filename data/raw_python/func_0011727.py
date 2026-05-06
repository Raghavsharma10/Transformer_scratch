def plot(ts, title=None, show=True):
    """Plot a Timeseries
    Args: 
      ts  Timeseries
      title  str
      show  bool whether to display the figure or just return a figure object  
      """
    ts = _remove_pi_crossings(ts)
    fig = plt.figure()
    ylabelprops = dict(rotation=0, 
                       horizontalalignment='right', 
                       verticalalignment='center', 
                       x=-0.01)
    if ts.ndim > 2: # multiple sim timeseries. collapse vars onto each subplot.
        num_subplots = ts.shape[ts.ndim - 1]
        if title is None:
            title = u'time series at each node'
        for i in range(num_subplots):
            ax = fig.add_subplot(num_subplots, 1, i+1)
            ax.plot(ts.tspan, ts[...,i])
            if ts.labels[-1] is not None:
                ax.set_ylabel(ts.labels[-1][i], **ylabelprops)
            else:
                ax.set_ylabel('node ' + str(i), **ylabelprops)
            plt.setp(ax.get_xticklabels(), visible=False)
        fig.axes[0].set_title(title)
        plt.setp(fig.axes[num_subplots-1].get_xticklabels(), visible=True)
        fig.axes[num_subplots-1].set_xlabel('time (s)')
    else: # single sim timeseries. show each variable separately.
        if ts.ndim is 1:
            ts = ts.reshape((-1, 1))
        num_ax = ts.shape[1]
        if title is None:
            title=u'time series'
        axprops = dict()
        if num_ax > 10:
            axprops['yticks'] = []
        colors = _get_color_list()
        for i in range(num_ax):
            rect = 0.1, 0.85*(num_ax - i - 1)/num_ax + 0.1, 0.8, 0.85/num_ax
            ax = fig.add_axes(rect, **axprops)
            ax.plot(ts.tspan, ts[...,i], color=colors[i % len(colors)])
            plt.setp(ax.get_xticklabels(), visible=False)
            if ts.labels[1] is not None:
                ax.set_ylabel(ts.labels[1][i], **ylabelprops)
        fig.axes[0].set_title(title)
        plt.setp(fig.axes[num_ax-1].get_xticklabels(), visible=True)
        fig.axes[num_ax-1].set_xlabel('time (s)')
    if show:
        fig.show()
    return fig