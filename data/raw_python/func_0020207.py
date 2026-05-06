def StatsToCSV(campaign, model='nPLD'):
    '''
    Generate the CSV file used in the search database for the documentation.
    '''
    statsfile = os.path.join(EVEREST_SRC, 'missions', 'k2',
                             'tables', 'c%02d_%s.cdpp' % (campaign, model))
    csvfile = os.path.join(os.path.dirname(EVEREST_SRC), 'docs',
                           'c%02d.csv' % campaign)
    epic, kp, cdpp6r, cdpp6, _, _, _, _, saturated = \
        np.loadtxt(statsfile, unpack=True, skiprows=2)

    with open(csvfile, 'w') as f:
        print('c%02d' % campaign, file=f)
        for i in range(len(epic)):
            print('%09d,%.3f,%.3f,%.3f,%d' % (epic[i], kp[i],
                                              cdpp6r[i], cdpp6[i],
                                              int(saturated[i])),
                                              file=f)