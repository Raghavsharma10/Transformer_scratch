def _Publish(campaign, subcampaign, strkwargs):
    '''
    The actual function that publishes a given campaign; this must
    be called from ``missions/k2/publish.pbs``.

    '''

    # Get kwargs from string
    kwargs = pickle.loads(strkwargs.replace('%%%', '\n').encode('utf-8'))

    # Check the cadence
    cadence = kwargs.get('cadence', 'lc')

    # Model wrapper
    m = FunctionWrapper(EverestModel, season=campaign, publish=True, **kwargs)

    # Set up our custom exception handler
    sys.excepthook = ExceptionHook

    # Initialize our multiprocessing pool
    with Pool() as pool:
        # Are we doing a subcampaign?
        if subcampaign != -1:
            campaign = campaign + 0.1 * subcampaign
        # Get all the stars
        stars = GetK2Campaign(campaign, epics_only=True, cadence=cadence)

        # Run
        pool.map(m, stars)