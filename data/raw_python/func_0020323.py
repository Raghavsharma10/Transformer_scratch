def GetK2Campaign(campaign, clobber=False, split=False,
                  epics_only=False, cadence='lc'):
    '''
    Return all stars in a given *K2* campaign.

    :param campaign: The *K2* campaign number. If this is an :py:class:`int`, \
           returns all targets in that campaign. If a :py:class:`float` in \
           the form :py:obj:`X.Y`, runs the :py:obj:`Y^th` decile of campaign \
           :py:obj:`X`.
    :param bool clobber: If :py:obj:`True`, download and overwrite existing \
           files. Default :py:obj:`False`
    :param bool split: If :py:obj:`True` and :py:obj:`campaign` is an \
           :py:class:`int`, returns each of the subcampaigns as a separate \
           list. Default :py:obj:`False`
    :param bool epics_only: If :py:obj:`True`, returns only the EPIC numbers. \
           If :py:obj:`False`, returns metadata associated with each target. \
           Default :py:obj:`False`
    :param str cadence: Long (:py:obj:`lc`) or short (:py:obj:`sc`) cadence? \
           Default :py:obj:`lc`.

    '''

    all = GetK2Stars(clobber=clobber)
    if int(campaign) in all.keys():
        all = all[int(campaign)]
    else:
        return []

    if cadence == 'sc':
        all = [a for a in all if a[3]]

    if epics_only:
        all = [a[0] for a in all]
    if type(campaign) is int or type(campaign) is np.int64:
        if not split:
            return all
        else:
            all_split = list(Chunks(all, len(all) // 10))

            # HACK: Sometimes we're left with a few targets
            # dangling at the end. Insert them back evenly
            # into the first few subcampaigns.
            if len(all_split) > 10:
                tmp1 = all_split[:10]
                tmp2 = all_split[10:]
                for n in range(len(tmp2)):
                    tmp1[n] = np.append(tmp1[n], tmp2[n])
                all_split = tmp1

            res = []
            for subcampaign in range(10):
                res.append(all_split[subcampaign])

            return res
    elif type(campaign) is float:
        x, y = divmod(campaign, 1)
        campaign = int(x)
        subcampaign = round(y * 10)
        return list(Chunks(all, len(all) // 10))[subcampaign]
    else:
        raise Exception('Argument `subcampaign` must be an `int` ' +
                        'or a `float` in the form `X.Y`')