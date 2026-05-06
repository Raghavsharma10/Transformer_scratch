def fetch(force=False):
    """Fetch and extract latest Life-Line version of Fiji is just ImageJ
    to *~/.bin*.

    Parameters
    ----------
    force : bool
        Force overwrite of existing Fiji in *~/.bin*.

    """
    try:
        # python 2
        from urllib2 import urlopen, HTTPError, URLError
    except ImportError:
        # python 3
        from urllib.request import urlopen, HTTPError, URLError

    if os.path.isdir(FIJI_ROOT) and not force:
        return
    elif not os.path.isdir(FIJI_ROOT):
        print('Fiji missing in %s' % FIJI_ROOT)

    if force:
        print('Deleting %s' % FIJI_ROOT)
        shutil.rmtree(FIJI_ROOT, ignore_errors=True)

    print('Downloading fiji from %s' % URL)
    try:
        req = urlopen(URL)
        try:
            size = int(req.info()['content-length'])
        except AttributeError:
            size = -1

        chunk = 512*1024
        fp = BytesIO()
        i = 0
        while 1:
            data = req.read(chunk)
            if not data:
                break
            fp.write(data)
            if size > 0:
                percent = fp.tell() // (size/100)
                msg = 'Downloaded %d percent      \r' % percent
            else:
                msg = 'Downloaded %d bytes\r' % fp.tell()
            sys.stdout.write(msg)
    except (HTTPError, URLError) as e:
        print('Error getting fiji: {}'.format(e))
        sys.exit(1)

    try:
        print('\nExtracting zip')
        z = ZipFile(fp)
        z.extractall(BIN_FOLDER)
        # move to Fiji-VERSION.app to easily check if it exists (upon fijibin upgrade)
        os.rename(EXTRACT_FOLDER, FIJI_ROOT)
    except (BadZipFile, IOError) as e:
        print('Error extracting zip: {}'.format(e))
        sys.exit(1)

    for path in BIN_NAMES.values():
        st = os.stat(path)
        os.chmod(path, st.st_mode | stat.S_IEXEC)