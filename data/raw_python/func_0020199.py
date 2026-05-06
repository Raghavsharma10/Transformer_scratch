def ShortCadenceStatistics(campaign=None, clobber=False, model='nPLD',
                           plot=True, **kwargs):
    '''
    Computes and plots the CDPP statistics comparison between short cadence
    and long cadence de-trended light curves

    :param campaign: The campaign number or list of campaign numbers. \
           Default is to plot all campaigns
    :param bool clobber: Overwrite existing files? Default :py:obj:`False`
    :param str model: The :py:obj:`everest` model name
    :param bool plot: Default :py:obj:`True`

    '''

    # Check campaign
    if campaign is None:
        campaign = np.arange(9)
    else:
        campaign = np.atleast_1d(campaign)

    # Update model name
    model = '%s.sc' % model

    # Compute the statistics
    for camp in campaign:
        sub = np.array(GetK2Campaign(
            camp, cadence='sc', epics_only=True), dtype=int)
        outfile = os.path.join(EVEREST_SRC, 'missions', 'k2',
                               'tables', 'c%02d_%s.cdpp' % (int(camp), model))
        if clobber or not os.path.exists(outfile):
            with open(outfile, 'w') as f:
                print("EPIC               Kp           Raw CDPP     " +
                      "Everest CDPP      Saturated", file=f)
                print("---------          ------       ---------    " +
                      "------------      ---------", file=f)
                all = GetK2Campaign(int(camp), cadence='sc')
                stars = np.array([s[0] for s in all], dtype=int)
                kpmgs = np.array([s[1] for s in all], dtype=float)
                for i, _ in enumerate(stars):
                    sys.stdout.write(
                        '\rProcessing target %d/%d...' % (i + 1, len(stars)))
                    sys.stdout.flush()
                    nf = os.path.join(EVEREST_DAT, 'k2', 'c%02d' % camp,
                                      ('%09d' % stars[i])[:4] + '00000',
                                      ('%09d' % stars[i])[4:], model + '.npz')
                    try:
                        data = np.load(nf)
                        print("{:>09d} {:>15.3f} {:>15.3f} {:>15.3f} {:>15d}".format(
                              stars[i], kpmgs[i], data['cdppr'][()], data['cdpp'][()],
                              int(data['saturated'])), file=f)
                    except:
                        print("{:>09d} {:>15.3f} {:>15.3f} {:>15.3f} {:>15d}".format(
                            stars[i], kpmgs[i], np.nan, np.nan, 0), file=f)
                print("")

    if not plot:
        return

    # Running lists
    xsat = []
    ysat = []
    xunsat = []
    yunsat = []
    xall = []
    yall = []
    epics = []

    # Plot
    for camp in campaign:

        # Load all stars
        sub = np.array(GetK2Campaign(
            camp, cadence='sc', epics_only=True), dtype=int)
        outfile = os.path.join(EVEREST_SRC, 'missions', 'k2',
                               'tables', 'c%02d_%s.cdpp' % (int(camp), model))
        epic, kp, cdpp6r, cdpp6, saturated = np.loadtxt(
            outfile, unpack=True, skiprows=2)
        epic = np.array(epic, dtype=int)
        saturated = np.array(saturated, dtype=int)

        # Get only stars in this subcamp
        inds = np.array([e in sub for e in epic])
        epic = epic[inds]
        kp = kp[inds]
        # HACK: camp 0 magnitudes are reported only to the nearest tenth,
        # so let's add a little noise to spread them out for nicer plotting
        kp = kp + 0.1 * (0.5 - np.random.random(len(kp)))
        cdpp6r = cdpp6r[inds]
        cdpp6 = cdpp6[inds]
        saturated = saturated[inds]
        sat = np.where(saturated == 1)
        unsat = np.where(saturated == 0)
        if not np.any([not np.isnan(x) for x in cdpp6]):
            continue

        # Get the long cadence stats
        compfile = os.path.join(EVEREST_SRC, 'missions', 'k2',
                                'tables', 'c%02d_%s.cdpp' % (int(camp),
                                                             model[:-3]))
        epic_1, _, _, cdpp6_1, _, _, _, _, saturated = np.loadtxt(
            compfile, unpack=True, skiprows=2)
        epic_1 = np.array(epic_1, dtype=int)
        inds = np.array([e in sub for e in epic_1])
        epic_1 = epic_1[inds]
        cdpp6_1 = cdpp6_1[inds]
        cdpp6_1 = sort_like(cdpp6_1, epic, epic_1)
        x = kp
        y = (cdpp6 - cdpp6_1) / cdpp6_1

        # Append to running lists
        xsat.extend(x[sat])
        ysat.extend(y[sat])
        xunsat.extend(x[unsat])
        yunsat.extend(y[unsat])
        xall.extend(x)
        yall.extend(y)
        epics.extend(epic)

    # Plot the equivalent of the Aigrain+16 figure
    fig, ax = pl.subplots(1)
    fig.canvas.set_window_title('K2 Short Cadence')

    ax.scatter(xunsat, yunsat, color='b', marker='.',
               alpha=0.35, zorder=-1, picker=True)
    ax.scatter(xsat, ysat, color='r', marker='.',
               alpha=0.35, zorder=-1, picker=True)
    ax.set_ylim(-1, 1)
    ax.set_xlim(8, 18)
    ax.axhline(0, color='gray', lw=2, zorder=-99, alpha=0.5)
    ax.axhline(0.5, color='gray', ls='--', lw=2, zorder=-99, alpha=0.5)
    ax.axhline(-0.5, color='gray', ls='--', lw=2, zorder=-99, alpha=0.5)
    ax.set_title(r'Short Versus Long Cadence', fontsize=18)
    ax.set_ylabel(r'Relative CDPP', fontsize=18)
    ax.set_xlabel('Kepler Magnitude', fontsize=18)

    # Bin the CDPP
    yall = np.array(yall)
    xall = np.array(xall)
    bins = np.arange(7.5, 18.5, 0.5)
    by = np.zeros_like(bins) * np.nan
    for b, bin in enumerate(bins):
        i = np.where((yall > -np.inf) & (yall < np.inf) &
                     (xall >= bin - 0.5) & (xall < bin + 0.5))[0]
        if len(i) > 10:
            by[b] = np.median(yall[i])
    ax.plot(bins[:9], by[:9], 'r--', lw=2)
    ax.plot(bins[8:], by[8:], 'k-', lw=2)

    # Pickable points
    Picker = StatsPicker([ax], [xall], [yall], epics, model=model[:-3],
                         compare_to=model[:-3], cadence='sc',
                         campaign=campaign)
    fig.canvas.mpl_connect('pick_event', Picker)

    # Show
    pl.show()