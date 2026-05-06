def plot_run_nlive(method_names, run_dict, **kwargs):
    """Plot the allocations of live points as a function of logX for the input
    sets of nested sampling runs of the type used in the dynamic nested
    sampling paper (Higson et al. 2019).
    Plots also include analytically calculated distributions of relative
    posterior mass and relative posterior mass remaining.

    Parameters
    ----------
    method_names: list of strs
    run_dict: dict of lists of nested sampling runs.
        Keys of run_dict must be method_names.
    logx_given_logl: function, optional
        For mapping points' logl values to logx values.
        If not specified the logx coordinates for each run are estimated using
        its numbers of live points.
    logl_given_logx: function, optional
        For calculating the relative posterior mass and posterior mass
        remaining at each logx coordinate.
    logx_min: float, optional
        Lower limit of logx axis. If not specified this is set to the lowest
        logx reached by any of the runs.
    ymax: bool, optional
        Maximum value for plot's nlive axis (yaxis).
    npoints: int, optional
        Number of points to have in the fgivenx plot grids.
    figsize: tuple, optional
        Size of figure in inches.
    post_mass_norm: str or None, optional
        Specify method_name for runs use form normalising the analytic
        posterior mass curve. If None, all runs are used.
    cum_post_mass_norm: str or None, optional
        Specify method_name for runs use form normalising the analytic
        cumulative posterior mass remaining curve. If None, all runs are used.

    Returns
    -------
    fig: matplotlib figure
    """
    logx_given_logl = kwargs.pop('logx_given_logl', None)
    logl_given_logx = kwargs.pop('logl_given_logx', None)
    logx_min = kwargs.pop('logx_min', None)
    ymax = kwargs.pop('ymax', None)
    npoints = kwargs.pop('npoints', 100)
    figsize = kwargs.pop('figsize', (6.4, 2))
    post_mass_norm = kwargs.pop('post_mass_norm', None)
    cum_post_mass_norm = kwargs.pop('cum_post_mass_norm', None)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    assert set(method_names) == set(run_dict.keys()), (
        'input method names=' + str(method_names) + ' do not match run_dict '
        'keys=' + str(run_dict.keys()))
    # Plotting
    # --------
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Reserve colors for certain common method_names so they are always the
    # same regardless of method_name order for consistency in the paper.
    linecolor_dict = {'standard': colors[2],
                      'dynamic $G=0$': colors[8],
                      'dynamic $G=1$': colors[9]}
    ax.set_prop_cycle('color', [colors[i] for i in [4, 1, 6, 0, 3, 5, 7]])
    integrals_dict = {}
    logx_min_list = []
    for method_name in method_names:
        integrals = np.zeros(len(run_dict[method_name]))
        for nr, run in enumerate(run_dict[method_name]):
            if 'logx' in run:
                logx = run['logx']
            elif logx_given_logl is not None:
                logx = logx_given_logl(run['logl'])
            else:
                logx = nestcheck.ns_run_utils.get_logx(
                    run['nlive_array'], simulate=False)
            logx_min_list.append(logx[-1])
            logx[0] = 0  # to make lines extend all the way to the end
            if nr == 0:
                # Label the first line and store it so we can access its color
                try:
                    line, = ax.plot(logx, run['nlive_array'], linewidth=1,
                                    label=method_name,
                                    color=linecolor_dict[method_name])
                except KeyError:
                    line, = ax.plot(logx, run['nlive_array'], linewidth=1,
                                    label=method_name)
            else:
                # Set other lines to same color and don't add labels
                ax.plot(logx, run['nlive_array'], linewidth=1,
                        color=line.get_color())
            # for normalising analytic weight lines
            integrals[nr] = -np.trapz(run['nlive_array'], x=logx)
        integrals_dict[method_name] = integrals[np.isfinite(integrals)]
    # if not specified, set logx min to the lowest logx reached by a run
    if logx_min is None:
        logx_min = np.asarray(logx_min_list).min()
    if logl_given_logx is not None:
        # Plot analytic posterior mass and cumulative posterior mass
        logx_plot = np.linspace(logx_min, 0, npoints)
        logl = logl_given_logx(logx_plot)
        # Remove any NaNs
        logx_plot = logx_plot[np.where(~np.isnan(logl))[0]]
        logl = logl[np.where(~np.isnan(logl))[0]]
        w_an = rel_posterior_mass(logx_plot, logl)
        # Try normalising the analytic distribution of posterior mass to have
        # the same area under the curve as the runs with dynamic_goal=1 (the
        # ones which we want to compare to it). If they are not available just
        # normalise it to the average area under all the runs (which should be
        # about the same if they have the same number of samples).
        w_an *= average_by_key(integrals_dict, post_mass_norm)
        ax.plot(logx_plot, w_an,
                linewidth=2, label='relative posterior mass',
                linestyle=':', color='k')
        # plot cumulative posterior mass
        w_an_c = np.cumsum(w_an)
        w_an_c /= np.trapz(w_an_c, x=logx_plot)
        # Try normalising the cumulative distribution of posterior mass to have
        # the same area under the curve as the runs with dynamic_goal=0 (the
        # ones which we want to compare to it). If they are not available just
        # normalise it to the average area under all the runs (which should be
        # about the same if they have the same number of samples).
        w_an_c *= average_by_key(integrals_dict, cum_post_mass_norm)
        ax.plot(logx_plot, w_an_c, linewidth=2, linestyle='--', dashes=(2, 3),
                label='posterior mass remaining', color='darkblue')
    ax.set_ylabel('number of live points')
    ax.set_xlabel(r'$\log X $')
    # set limits
    if ymax is not None:
        ax.set_ylim([0, ymax])
    else:
        ax.set_ylim(bottom=0)
    ax.set_xlim([logx_min, 0])
    ax.legend()
    return fig