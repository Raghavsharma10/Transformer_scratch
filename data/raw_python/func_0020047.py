def InjectionStatus(campaign=range(18), model='nPLD', purge=False,
                    depths=[0.01, 0.001, 0.0001], **kwargs):
    '''
    Shows the progress of the injection de-trending runs for
    the specified campaign(s).

    '''

    if not hasattr(campaign, '__len__'):
        if type(campaign) is int:
            # Return the subcampaigns
            all_stars = [s for s in GetK2Campaign(
                campaign, split=True, epics_only=True)]
            campaign = [campaign + 0.1 * n for n in range(10)]
        else:
            all_stars = [[s for s in GetK2Campaign(campaign, epics_only=True)]]
            campaign = [campaign]
    else:
        all_stars = [[s for s in GetK2Campaign(
            c, epics_only=True)] for c in campaign]
    print("CAMP      MASK       DEPTH     TOTAL      DONE     ERRORS")
    print("----      ----       -----     -----      ----     ------")
    for c, stars in zip(campaign, all_stars):
        if len(stars) == 0:
            continue
        done = [[0 for d in depths], [0 for d in depths]]
        err = [[0 for d in depths], [0 for d in depths]]
        total = len(stars)
        if os.path.exists(os.path.join(EVEREST_DAT, 'k2', 'c%02d' % c)):
            path = os.path.join(EVEREST_DAT, 'k2', 'c%02d' % c)
            for folder in os.listdir(path):
                for subfolder in os.listdir(os.path.join(path, folder)):
                    ID = int(folder[:4] + subfolder)
                    for m, mask in enumerate(['U', 'M']):
                        for d, depth in enumerate(depths):
                            if os.path.exists(
                                os.path.join(
                                    EVEREST_DAT, 'k2', 'c%02d' % c, folder,
                                    subfolder, '%s_Inject_%s%g.npz' %
                                    (model, mask, depth))):
                                done[m][d] += 1
                            elif os.path.exists(
                                    os.path.join(
                                        EVEREST_DAT, 'k2', 'c%02d' % c, folder,
                                        subfolder, '%s_Inject_%s%g.err' %
                                        (model, mask, depth))):
                                err[m][d] += 1
        for d, depth in enumerate(depths):
            for m, mask in enumerate(['F', 'T']):
                if done[m][d] == total:
                    color = GREEN
                else:
                    color = BLACK
                if err[m][d] > 0:
                    errcolor = RED
                else:
                    errcolor = ''
                if type(c) is int:
                    print("%s{:>4d}{:>8s}{:>14g}{:>10d}{:>10d}%s{:>9d}\033[0m".format(
                        c, mask, depth, total, done[m][d], err[m][d]) % (color, errcolor))
                else:
                    print("%s{:>4.1f}{:>8s}{:>14g}{:>10d}{:>10d}%s{:>9d}\033[0m".format(
                        c, mask, depth, total, done[m][d], err[m][d]) % (color, errcolor))