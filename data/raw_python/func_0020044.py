def Run(campaign=0, EPIC=None, nodes=5, ppn=12, walltime=100,
        mpn=None, email=None, queue=None, **kwargs):
    '''
    Submits a cluster job to compute and plot data for all targets
    in a given campaign.

    :param campaign: The K2 campaign number. If this is an :py:class:`int`, \
           returns all targets in that campaign. If a :py:class:`float` \
           in the form `X.Y`, runs the `Y^th` decile of campaign `X`.
    :param str queue: The queue to submit to. Default `None` (default queue)
    :param str email: The email to send job status notifications to. \
           Default `None`
    :param int walltime: The number of hours to request. Default `100`
    :param int nodes: The number of nodes to request. Default `5`
    :param int ppn: The number of processors per node to request. Default `12`
    :param int mpn: Memory per node in gb to request. Default no setting.

    '''

    # Figure out the subcampaign
    if type(campaign) is int:
        subcampaign = -1
    elif type(campaign) is float:
        x, y = divmod(campaign, 1)
        campaign = int(x)
        subcampaign = round(y * 10)

    # DEV hack: limit backfill jobs to 10 hours
    if EVEREST_DEV and (queue == 'bf'):
        walltime = min(10, walltime)

    # Convert kwargs to string. This is really hacky. Pickle creates an array
    # of bytes, which we must convert into a regular string to pass to the pbs
    # script and then back into python. Decoding the bytes isn't enough, since
    # we have pesky escaped characters such as newlines that don't behave well
    # when passing this string around. My braindead hack is to replace newlines
    # with '%%%', then undo the replacement when reading the kwargs. This works
    # for most cases, but sometimes pickle creates a byte array that can't be
    # decoded into utf-8; this happens when trying to pass numpy arrays around,
    # for instance. This needs to be fixed in the future, but for now we'll
    # restrict the kwargs to be ints, floats, lists, and strings.
    try:
        strkwargs = pickle.dumps(kwargs, 0).decode(
            'utf-8').replace('\n', '%%%')
    except UnicodeDecodeError:
        raise ValueError('Unable to pickle `kwargs`. Currently the `kwargs` ' +
                         'values may only be `int`s, `float`s, `string`s, ' +
                         '`bool`s, or lists of these.')

    # Submit the cluster job
    pbsfile = os.path.join(EVEREST_SRC, 'missions', 'k2', 'run.pbs')
    if mpn is not None:
        str_n = 'nodes=%d:ppn=%d,feature=%dcore,mem=%dgb' % (
            nodes, ppn, ppn, mpn * nodes)
    else:
        str_n = 'nodes=%d:ppn=%d,feature=%dcore' % (nodes, ppn, ppn)
    str_w = 'walltime=%d:00:00' % walltime
    str_v = "EVEREST_DAT=%s,NODES=%d," % (EVEREST_DAT, nodes) + \
            "EPIC=%d," % (0 if EPIC is None else EPIC) + \
            "CAMPAIGN=%d,SUBCAMPAIGN=%d,STRKWARGS='%s'" % \
            (campaign, subcampaign, strkwargs)

    if EPIC is None:
        if subcampaign == -1:
            str_name = 'c%02d' % campaign
        else:
            str_name = 'c%02d.%d' % (campaign, subcampaign)
    else:
        str_name = 'EPIC%d' % EPIC
    str_out = os.path.join(EVEREST_DAT, 'k2', str_name + '.log')
    qsub_args = ['qsub', pbsfile,
                 '-v', str_v,
                 '-o', str_out,
                 '-j', 'oe',
                 '-N', str_name,
                 '-l', str_n,
                 '-l', str_w]
    if email is not None:
        qsub_args.append(['-M', email, '-m', 'ae'])
    if queue is not None:
        qsub_args += ['-q', queue]

    # Now we submit the job
    print("Submitting the job...")
    subprocess.call(qsub_args)