def InjectionStatistics(campaign=0, clobber=False, model='nPLD', plot=True,
                        show=True, **kwargs):
    '''
    Computes and plots the statistics for injection/recovery tests.

    :param int campaign: The campaign number. Default 0
    :param str model: The :py:obj:`everest` model name
    :param bool plot: Default :py:obj:`True`
    :param bool show: Show the plot? Default :py:obj:`True`. \
           If :py:obj:`False`, returns the `fig, ax` instances.
    :param bool clobber: Overwrite existing files? Default :py:obj:`False`

    '''

    # Compute the statistics
    stars = GetK2Campaign(campaign, epics_only=True)
    if type(campaign) is int:
        outfile = os.path.join(EVEREST_SRC, 'missions', 'k2',
                               'tables', 'c%02d_%s.inj' % (campaign, model))
    else:
        outfile = os.path.join(EVEREST_SRC, 'missions', 'k2',
                               'tables', 'c%04.1f_%s.inj' % (campaign, model))
    if clobber or not os.path.exists(outfile):
        with open(outfile, 'w') as f:
            print("EPIC         Depth         UControl      URecovered"+
                  "    MControl      MRecovered", file=f)
            print("---------    ----------    ----------    ----------"+
                  "    ----------    ----------", file=f)
            for i, _ in enumerate(stars):
                sys.stdout.write('\rProcessing target %d/%d...' %
                                 (i + 1, len(stars)))
                sys.stdout.flush()
                path = os.path.join(EVEREST_DAT, 'k2', 'c%02d' % int(campaign),
                                    ('%09d' % stars[i])[:4] + '00000',
                                    ('%09d' % stars[i])[4:])

                # Loop over all depths
                for depth in [0.01, 0.001, 0.0001]:

                    try:

                        # Unmasked
                        data = np.load(os.path.join(
                            path, '%s_Inject_U%g.npz' % (model, depth)))
                        assert depth == data['inject'][()]['depth'], ""
                        ucontrol = data['inject'][()]['rec_depth_control']
                        urecovered = data['inject'][()]['rec_depth']

                        # Masked
                        data = np.load(os.path.join(
                            path, '%s_Inject_M%g.npz' % (model, depth)))
                        assert depth == data['inject'][()]['depth'], ""
                        mcontrol = data['inject'][()]['rec_depth_control']
                        mrecovered = data['inject'][()]['rec_depth']

                        # Log it
                        print("{:>09d} {:>13.8f} {:>13.8f} {:>13.8f} {:>13.8f} {:>13.8f}".format(
                              stars[i], depth, ucontrol, urecovered, mcontrol,
                              mrecovered), file=f)

                    except:
                        pass

            print("")

    if plot:

        # Load the statistics
        try:
            epic, depth, ucontrol, urecovered, mcontrol, mrecovered = \
                np.loadtxt(outfile, unpack=True, skiprows=2)
        except ValueError:
            raise Exception("No targets to plot.")

        # Normalize to the injected depth
        ucontrol /= depth
        urecovered /= depth
        mcontrol /= depth
        mrecovered /= depth

        # Set up the plot
        fig, ax = pl.subplots(3, 2, figsize=(9, 12))
        fig.subplots_adjust(hspace=0.29)
        ax[0, 0].set_title(r'Unmasked', fontsize=18)
        ax[0, 1].set_title(r'Masked', fontsize=18)
        ax[0, 0].set_ylabel(
            r'$D_0 = 10^{-2}$', rotation=90, fontsize=18, labelpad=10)
        ax[1, 0].set_ylabel(
            r'$D_0 = 10^{-3}$', rotation=90, fontsize=18, labelpad=10)
        ax[2, 0].set_ylabel(
            r'$D_0 = 10^{-4}$', rotation=90, fontsize=18, labelpad=10)

        # Define some useful stuff for plotting
        depths = [1e-2, 1e-3, 1e-4]
        ranges = [(0.75, 1.25), (0.5, 1.5), (0., 2.)]
        nbins = [30, 30, 20]
        ymax = [0.4, 0.25, 0.16]
        xticks = [[0.75, 0.875, 1., 1.125, 1.25], [
            0.5, 0.75, 1., 1.25, 1.5], [0., 0.5, 1., 1.5, 2.0]]

        # Plot
        for i in range(3):

            # Indices for this plot
            idx = np.where(depth == depths[i])

            for j, control, recovered in zip([0, 1], [ucontrol[idx],
                                                      mcontrol[idx]],
                                                     [urecovered[idx],
                                                      mrecovered[idx]]):

                # Control
                ax[i, j].hist(control, bins=nbins[i], range=ranges[i],
                              color='r', histtype='step',
                              weights=np.ones_like(control) / len(control))

                # Recovered
                ax[i, j].hist(recovered, bins=nbins[i], range=ranges[i],
                              color='b', histtype='step',
                              weights=np.ones_like(recovered) / len(recovered))

                # Indicate center
                ax[i, j].axvline(1., color='k', ls='--')

                # Indicate the fraction above and below
                if len(recovered):
                    au = len(np.where(recovered > ranges[i][1])[
                             0]) / len(recovered)
                    al = len(np.where(recovered < ranges[i][0])[
                             0]) / len(recovered)
                    ax[i, j].annotate('%.2f' % al, xy=(0.01, 0.93),
                                      xycoords='axes fraction',
                                      xytext=(0.1, 0.93), ha='left',
                                      va='center', color='b',
                                      arrowprops=dict(arrowstyle="->",
                                      color='b'))
                    ax[i, j].annotate('%.2f' % au, xy=(0.99, 0.93),
                                      xycoords='axes fraction',
                                      xytext=(0.9, 0.93), ha='right',
                                      va='center', color='b',
                                      arrowprops=dict(arrowstyle="->",
                                      color='b'))
                if len(control):
                    cu = len(np.where(control > ranges[i][1])[
                             0]) / len(control)
                    cl = len(np.where(control < ranges[i][0])[
                             0]) / len(control)
                    ax[i, j].annotate('%.2f' % cl, xy=(0.01, 0.86),
                                      xycoords='axes fraction',
                                      xytext=(0.1, 0.86), ha='left',
                                      va='center', color='r',
                                      arrowprops=dict(arrowstyle="->",
                                      color='r'))
                    ax[i, j].annotate('%.2f' % cu, xy=(0.99, 0.86),
                                      xycoords='axes fraction',
                                      xytext=(0.9, 0.86), ha='right',
                                      va='center', color='r',
                                      arrowprops=dict(arrowstyle="->",
                                      color='r'))

                # Indicate the median
                if len(recovered):
                    ax[i, j].annotate('M = %.2f' % np.median(recovered),
                                      xy=(0.35, 0.5), ha='right',
                                      xycoords='axes fraction', color='b',
                                      fontsize=16)
                if len(control):
                    ax[i, j].annotate('M = %.2f' % np.median(control),
                                      xy=(0.65, 0.5), ha='left',
                                      xycoords='axes fraction',
                                      color='r', fontsize=16)

                # Tweaks
                ax[i, j].set_xticks(xticks[i])
                ax[i, j].set_xlim(xticks[i][0], xticks[i][-1])
                ax[i, j].set_ylim(-0.005, ymax[i])
                ax[i, j].set_xlabel(r'$D/D_0$', fontsize=16)

                ax[i, j].get_yaxis().set_major_locator(MaxNLocator(5))
                for tick in ax[i, j].get_xticklabels() + \
                        ax[i, j].get_yticklabels():
                    tick.set_fontsize(14)

        if show:
            pl.show()
        else:
            return fig, ax