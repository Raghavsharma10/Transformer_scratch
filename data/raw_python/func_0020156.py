def DownloadFile(ID, season=None, mission='k2', cadence='lc',
                 filename=None, clobber=False):
    '''
    Download a given :py:mod:`everest` file from MAST.

    :param str mission: The mission name. Default `k2`
    :param str cadence: The light curve cadence. Default `lc`
    :param str filename: The name of the file to download. Default \
           :py:obj:`None`, in which case the default \
           FITS file is retrieved.
    :param bool clobber: If :py:obj:`True`, download and overwrite \
           existing files. Default :py:obj:`False`

    '''

    # Get season
    if season is None:
        season = getattr(missions, mission).Season(ID)
    if hasattr(season, '__len__'):
        raise AttributeError(
            "Please choose a `season` for this target: %s." % season)
    if season is None:
        if getattr(missions, mission).ISTARGET(ID):
            raise ValueError('Target not found in local database. ' +
                             'Run `everest.Search(%d)` for more information.'
                             % ID)
        else:
            raise ValueError('Invalid target ID.')
    path = getattr(missions, mission).TargetDirectory(ID, season)
    relpath = getattr(missions, mission).TargetDirectory(
        ID, season, relative=True)
    if filename is None:
        filename = getattr(missions, mission).FITSFile(ID, season, cadence)

    # Check if file exists
    if not os.path.exists(path):
        os.makedirs(path)
    elif os.path.exists(os.path.join(path, filename)) and not clobber:
        log.info('Found cached file.')
        return os.path.join(path, filename)

    # Get file URL
    log.info('Downloading the file...')
    fitsurl = getattr(missions, mission).FITSUrl(ID, season)
    if not fitsurl.endswith('/'):
        fitsurl += '/'

    # Download the data
    r = urllib.request.Request(fitsurl + filename)
    try:
        handler = urllib.request.urlopen(r)
        code = handler.getcode()
    except (urllib.error.HTTPError, urllib.error.URLError):
        code = 0
    if int(code) == 200:

        # Read the data
        data = handler.read()

        # Atomically save to disk
        f = NamedTemporaryFile("wb", delete=False)
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
        f.close()
        shutil.move(f.name, os.path.join(path, filename))

    else:

        # Something went wrong!
        log.error("Error code {0} for URL '{1}'".format(
            code, fitsurl + filename))

        # If the files can be accessed by `ssh`, let's try that
        # (development version only!)
        if EVEREST_FITS is None:
            raise Exception("Unable to locate the file.")

        # Get the url
        inpath = os.path.join(EVEREST_FITS, relpath, filename)
        outpath = os.path.join(path, filename)

        # Download the data
        log.info("Accessing file via `scp`...")
        subprocess.call(['scp', inpath, outpath])

    # Success?
    if os.path.exists(os.path.join(path, filename)):
        return os.path.join(path, filename)
    else:
        raise Exception("Unable to download the file." +
                        "Run `everest.Search(%d)` to troubleshoot." % ID)