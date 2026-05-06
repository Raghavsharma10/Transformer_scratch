def plot_network_results(network, ds=None, mean=0, std=1, title='', show=True, save=True):
    """Identical to plot_trainer except `network` and `ds` must be provided separately"""
    df = sim_network(network=network, ds=ds, mean=mean, std=std)
    df.plot()
    plt.xlabel('Date')
    plt.ylabel('Threshold (kW)')
    plt.title(title)

    if show:
        try:
            # ipython notebook overrides plt.show and doesn't have a block kwarg
            plt.show(block=False)
        except TypeError:
            plt.show()
    if save:
        filename = 'ann_performance_for_{0}.png'.format(title).replace(' ', '_')
        if isinstance(save, basestring) and os.path.isdir(save):
            filename = os.path.join(save, filename)
        plt.savefig(filename)
    if not show:
        plt.clf()

    return network, mean, std