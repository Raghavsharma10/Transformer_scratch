def _Download(campaign, subcampaign):
    '''
    Download all stars from a given campaign. This is
    called from ``missions/k2/download.pbs``

    '''

    # Are we doing a subcampaign?
    if subcampaign != -1:
        campaign = campaign + 0.1 * subcampaign
    # Get all star IDs for this campaign
    stars = [s[0] for s in GetK2Campaign(campaign)]
    nstars = len(stars)
    # Download the TPF data for each one
    for i, EPIC in enumerate(stars):
        print("Downloading data for EPIC %d (%d/%d)..." %
              (EPIC, i + 1, nstars))
        if not os.path.exists(os.path.join(EVEREST_DAT, 'k2',
                                           'c%02d' % int(campaign),
                                           ('%09d' % EPIC)[:4] + '00000',
                                           ('%09d' % EPIC)[4:],
                                           'data.npz')):
            try:
                GetData(EPIC, season=campaign, download_only=True)
            except KeyboardInterrupt:
                sys.exit()
            except:
                # Some targets could be corrupted...
                print("ERROR downloading EPIC %d." % EPIC)
                exctype, value, tb = sys.exc_info()
                for line in traceback.format_exception_only(exctype, value):
                    ln = line.replace('\n', '')
                    print(ln)
                continue