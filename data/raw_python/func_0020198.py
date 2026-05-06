def PlanetStatistics(model='nPLD', compare_to='k2sff', **kwargs):
    '''
    Computes and plots the CDPP statistics comparison between `model` and
    `compare_to` for all known K2 planets.

    :param str model: The :py:obj:`everest` model name
    :param str compare_to: The :py:obj:`everest` model name or \
           other K2 pipeline name

    '''

    # Load all planet hosts
    f = os.path.join(EVEREST_SRC, 'missions', 'k2', 'tables', 'planets.tsv')
    epic, campaign, kp, _, _, _, _, _, _ = np.loadtxt(
        f, unpack=True, skiprows=2)
    epic = np.array(epic, dtype=int)
    campaign = np.array(campaign, dtype=int)
    cdpp = np.zeros(len(epic))
    saturated = np.zeros(len(epic), dtype=int)
    cdpp_1 = np.zeros(len(epic))

    # Get the stats
    for c in set(campaign):

        # Everest model
        f = os.path.join(EVEREST_SRC, 'missions', 'k2',
                         'tables', 'c%02d_%s.cdpp' % (int(c), model))
        e0, _, _, c0, _, _, _, _, s0 = np.loadtxt(f, unpack=True, skiprows=2)
        for i, e in enumerate(epic):
            if e in e0:
                j = np.argmax(e0 == e)
                cdpp[i] = c0[j]
                saturated[i] = s0[j]

        # Comparison model
        f = os.path.join(EVEREST_SRC, 'missions', 'k2', 'tables',
                         'c%02d_%s.cdpp' % (int(c), compare_to.lower()))
        if not os.path.exists(f):
            continue
        if compare_to.lower() in ['everest1', 'k2sff', 'k2sc']:
            e1, c1 = np.loadtxt(f, unpack=True, skiprows=2)
        else:
            e1, _, _, c1, _, _, _, _, _ = np.loadtxt(
                f, unpack=True, skiprows=2)
        for i, e in enumerate(epic):
            if e in e1:
                j = np.argmax(e1 == e)
                cdpp_1[i] = c1[j]

    sat = np.where(saturated == 1)
    unsat = np.where(saturated == 0)

    # Plot the equivalent of the Aigrain+16 figure
    fig, ax = pl.subplots(1)
    fig.canvas.set_window_title(
        'K2 Planet Hosts: %s versus %s' % (model, compare_to))
    x = kp
    y = (cdpp - cdpp_1) / cdpp_1
    ax.scatter(x[unsat], y[unsat], color='b', marker='.',
               alpha=0.5, zorder=-1, picker=True)
    ax.scatter(x[sat], y[sat], color='r', marker='.',
               alpha=0.5, zorder=-1, picker=True)
    ax.set_ylim(-1, 1)
    ax.set_xlim(8, 18)
    ax.axhline(0, color='gray', lw=2, zorder=-99, alpha=0.5)
    ax.axhline(0.5, color='gray', ls='--', lw=2, zorder=-99, alpha=0.5)
    ax.axhline(-0.5, color='gray', ls='--', lw=2, zorder=-99, alpha=0.5)
    ax.set_title(r'K2 Planet Hosts', fontsize=18)
    ax.set_ylabel(r'Relative CDPP', fontsize=18)
    ax.set_xlabel('Kepler Magnitude', fontsize=18)

    # Pickable points
    Picker = StatsPicker([ax], [kp], [y], epic,
                         model=model, compare_to=compare_to)
    fig.canvas.mpl_connect('pick_event', Picker)

    # Show
    pl.show()