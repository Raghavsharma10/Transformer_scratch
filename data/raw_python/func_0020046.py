def Status(season=range(18), model='nPLD', purge=False, injection=False,
           cadence='lc', **kwargs):
    '''
    Shows the progress of the de-trending runs for the specified campaign(s).

    '''

    # Mission compatibility
    campaign = season

    # Injection?
    if injection:
        return InjectionStatus(campaign=campaign, model=model,
                               purge=purge, **kwargs)

    # Cadence
    if cadence == 'sc':
        model = '%s.sc' % model

    if not hasattr(campaign, '__len__'):
        if type(campaign) is int:
            # Return the subcampaigns
            all_stars = [s for s in GetK2Campaign(
                campaign, split=True, epics_only=True, cadence=cadence)]
            campaign = [campaign + 0.1 * n for n in range(10)]
        else:
            all_stars = [[s for s in GetK2Campaign(
                campaign, epics_only=True, cadence=cadence)]]
            campaign = [campaign]
    else:
        all_stars = [[s for s in GetK2Campaign(
            c, epics_only=True, cadence=cadence)] for c in campaign]

    print("CAMP      TOTAL      DOWNLOADED    PROCESSED      FITS    ERRORS")
    print("----      -----      ----------    ---------      ----    ------")
    for c, stars in zip(campaign, all_stars):
        if len(stars) == 0:
            continue
        down = 0
        proc = 0
        err = 0
        fits = 0
        bad = []
        remain = []
        total = len(stars)
        if os.path.exists(os.path.join(EVEREST_DAT, 'k2', 'c%02d' % c)):
            path = os.path.join(EVEREST_DAT, 'k2', 'c%02d' % c)
            for folder in [f for f in os.listdir(path) if f.endswith('00000')]:
                for subfolder in os.listdir(os.path.join(path, folder)):
                    ID = int(folder[:4] + subfolder)
                    if ID in stars:
                        if os.path.exists(os.path.join(EVEREST_DAT,
                                                       'k2', 'c%02d' % c,
                                                       folder,
                                                       subfolder, 'data.npz')):
                            down += 1
                        if os.path.exists(os.path.join(EVEREST_DAT, 'k2',
                                                       'c%02d' % c, folder,
                                                       subfolder, FITSFile(
                                                         ID, c,
                                                         cadence=cadence))):
                            fits += 1
                        if os.path.exists(os.path.join(EVEREST_DAT, 'k2',
                                                       'c%02d' % c, folder,
                                                       subfolder,
                                                       model + '.npz')):
                            proc += 1
                        elif os.path.exists(os.path.join(EVEREST_DAT, 'k2',
                                                         'c%02d' % c, folder,
                                                         subfolder,
                                                         model + '.err')):
                            err += 1
                            bad.append(folder[:4] + subfolder)
                            if purge:
                                os.remove(os.path.join(
                                    EVEREST_DAT, 'k2', 'c%02d' % c,
                                    folder, subfolder, model + '.err'))
                        else:
                            remain.append(folder[:4] + subfolder)
        if proc == total:
            cc = ct = cp = ce = GREEN
            cd = BLACK if down < total else GREEN
        else:
            cc = BLACK
            ct = BLACK
            cd = BLACK if down < total else BLUE
            cp = BLACK if proc < down or proc == 0 else BLUE
            ce = RED if err > 0 else BLACK
        cf = BLACK if fits < total else GREEN
        if type(c) is int:
            print("%s{:>4d}   \033[0m%s{:>8d}\033[0m%s{:>16d}\033[0m%s{:>13d}\033[0m%s{:>10d}\033[0m%s{:>10d}\033[0m".format(c, total, down, proc, fits, err)
                  % (cc, ct, cd, cp, cf, ce))
        else:
            print("%s{:>4.1f}   \033[0m%s{:>8d}\033[0m%s{:>16d}\033[0m%s{:>13d}\033[0m%s{:>10d}\033[0m%s{:>10d}\033[0m".format(c, total, down, proc, fits, err)
                  % (cc, ct, cd, cp, cf, ce))
        if len(remain) <= 25 and len(remain) > 0 and len(campaign) == 1:
            remain.extend(["         "] * (4 - (len(remain) % 4)))
            print()
            for A, B, C, D in zip(remain[::4], remain[1::4],
                                  remain[2::4], remain[3::4]):
                if A == remain[0]:
                    print("REMAIN:  %s   %s   %s   %s" % (A, B, C, D))
                    print()
                else:
                    print("         %s   %s   %s   %s" % (A, B, C, D))
                    print()
        if len(bad) and len(campaign) == 1:
            bad.extend(["         "] * (4 - (len(bad) % 4)))
            print()
            for A, B, C, D in zip(bad[::4], bad[1::4], bad[2::4], bad[3::4]):
                if A == bad[0]:
                    print("ERRORS:  %s   %s   %s   %s" % (A, B, C, D))
                    print()
                else:
                    print("         %s   %s   %s   %s" % (A, B, C, D))
                    print()