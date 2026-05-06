def Download(campaign=0, queue='build', email=None, walltime=8, **kwargs):
    '''
    Submits a cluster job to the build queue to download all TPFs for a given
    campaign.

    :param int campaign: The `K2` campaign to run
    :param str queue: The name of the queue to submit to. Default `build`
    :param str email: The email to send job status notifications to. \
           Default `None`
    :param int walltime: The number of hours to request. Default `8`

    '''

    # Figure out the subcampaign
    if type(campaign) is int:
        subcampaign = -1
    elif type(campaign) is float:
        x, y = divmod(campaign, 1)
        campaign = int(x)
        subcampaign = round(y * 10)
    # Submit the cluster job
    pbsfile = os.path.join(EVEREST_SRC, 'missions', 'k2', 'download.pbs')
    str_w = 'walltime=%d:00:00' % walltime
    str_v = 'EVEREST_DAT=%s,CAMPAIGN=%d,SUBCAMPAIGN=%d' % (
        EVEREST_DAT, campaign, subcampaign)
    if subcampaign == -1:
        str_name = 'download_c%02d' % campaign
    else:
        str_name = 'download_c%02d.%d' % (campaign, subcampaign)
    str_out = os.path.join(EVEREST_DAT, 'k2', str_name + '.log')
    qsub_args = ['qsub', pbsfile,
                 '-q', queue,
                 '-v', str_v,
                 '-o', str_out,
                 '-j', 'oe',
                 '-N', str_name,
                 '-l', str_w]
    if email is not None:
        qsub_args.append(['-M', email, '-m', 'ae'])
    # Now we submit the job
    print("Submitting the job...")
    subprocess.call(qsub_args)