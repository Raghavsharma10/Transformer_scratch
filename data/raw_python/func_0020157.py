def DVS(ID, season=None, mission='k2', clobber=False,
        cadence='lc', model='nPLD'):
    '''
    Show the data validation summary (DVS) for a given target.

    :param str mission: The mission name. Default `k2`
    :param str cadence: The light curve cadence. Default `lc`
    :param bool clobber: If :py:obj:`True`, download and overwrite \
           existing files. Default :py:obj:`False`

    '''

    # Get season
    if season is None:
        season = getattr(missions, mission).Season(ID)
    if hasattr(season, '__len__'):
        raise AttributeError(
            "Please choose a `season` for this target: %s." % season)

    # Get file name
    if model == 'nPLD':
        filename = getattr(missions, mission).DVSFile(ID, season, cadence)
    else:
        if cadence == 'sc':
            filename = model + '.sc.pdf'
        else:
            filename = model + '.pdf'

    file = DownloadFile(ID, season=season,
                        mission=mission,
                        filename=filename,
                        clobber=clobber)

    try:
        if platform.system().lower().startswith('darwin'):
            subprocess.call(['open', file])
        elif os.name == 'nt':
            os.startfile(file)
        elif os.name == 'posix':
            subprocess.call(['xdg-open', file])
        else:
            raise Exception("")
    except:
        log.info("Unable to open the pdf. Try opening it manually:")
        log.info(file)