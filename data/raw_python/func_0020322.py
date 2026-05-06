def GetK2Stars(clobber=False):
    '''
    Download and return a :py:obj:`dict` of all *K2* stars organized by
    campaign. Saves each campaign to a `.stars` file in the
    `everest/missions/k2/tables` directory.

    :param bool clobber: If :py:obj:`True`, download and overwrite \
           existing files. Default :py:obj:`False`

    .. note:: The keys of the dictionary returned by this function are the \
              (integer) numbers of each campaign. Each item in the \
              :py:obj:`dict` is a list of the targets in the corresponding \
              campaign, and each item in that list is in turn a list of the \
              following: **EPIC number** (:py:class:`int`), \
              **Kp magnitude** (:py:class:`float`), **CCD channel number** \
              (:py:class:`int`), and **short cadence available** \
              (:py:class:`bool`).

    '''

    # Download
    if clobber:
        print("Downloading K2 star list...")
        stars = kplr_client.k2_star_info()
        print("Writing star list to disk...")
        for campaign in stars.keys():
            if not os.path.exists(os.path.join(EVEREST_SRC, 'missions',
                                               'k2', 'tables')):
                os.makedirs(os.path.join(
                    EVEREST_SRC, 'missions', 'k2', 'tables'))
            with open(os.path.join(EVEREST_SRC, 'missions', 'k2', 'tables',
                                   'c%02d.stars' % campaign), 'w') as f:
                for star in stars[campaign]:
                    print(",".join([str(s) for s in star]), file=f)

    # Return
    res = {}
    for campaign in K2_CAMPAIGNS:
        f = os.path.join(EVEREST_SRC, 'missions', 'k2',
                         'tables', 'c%02d.stars' % campaign)
        if os.path.exists(f):
            with open(f, 'r') as file:
                lines = file.readlines()
                if len(lines[0].split(',')) == 4:
                    # EPIC number, Kp magnitude, channel number,
                    # short cadence available?
                    stars = [[int(l.split(',')[0]),
                              _float(l.split(',')[1]),
                              int(l.split(',')[2]),
                              eval(l.split(',')[3])] for l in lines]
                else:
                    stars = [[int(l), np.nan, -1, None] for l in lines]
            res.update({campaign: stars})

    return res