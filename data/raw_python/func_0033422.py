def load(path, filetype=None, as_df=False, retries=None,
         _oid=None, quiet=False, **kwargs):
    '''Load multiple files from various file types automatically.

    Supports glob paths, eg::

        path = 'data/*.csv'

    Filetypes are autodetected by common extension strings.

    Currently supports loadings from:
        * csv (pd.read_csv)
        * json (pd.read_json)

    :param path: path to config json file
    :param filetype: override filetype autodetection
    :param kwargs: additional filetype loader method kwargs
    '''
    is_true(HAS_PANDAS, "`pip install pandas` required")
    set_oid = set_oid_func(_oid)

    # kwargs are for passing ftype load options (csv.delimiter, etc)
    # expect the use of globs; eg, file* might result in fileN (file1,
    # file2, file3), etc
    if not isinstance(path, basestring):
        # assume we're getting a raw dataframe
        objects = path
        if not isinstance(objects, pd.DataFrame):
            raise ValueError("loading raw values must be DataFrames")
    elif re.match('https?://', path):
        logger.debug('Saving %s to tmp file' % path)
        _path = urlretrieve(path, retries)
        logger.debug('%s saved to tmp file: %s' % (path, _path))
        try:
            objects = load_file(_path, filetype, **kwargs)
        finally:
            remove_file(_path)
    else:
        path = re.sub('^file://', '', path)
        path = os.path.expanduser(path)
        # assume relative to cwd if not already absolute path
        path = path if os.path.isabs(path) else pjoin(os.getcwd(), path)
        files = sorted(glob.glob(os.path.expanduser(path)))
        if not files:
            raise IOError("failed to load: %s" % path)
        # buid up a single dataframe by concatting
        # all globbed files together
        objects = []
        [objects.extend(load_file(ds, filetype, **kwargs))
            for ds in files]

    if is_empty(objects, except_=False) and not quiet:
        raise RuntimeError("no objects extracted!")
    else:
        logger.debug("Data loaded successfully from %s" % path)

    if set_oid:
        # set _oids, if we have a _oid generator func defined
        objects = [set_oid(o) for o in objects]

    if as_df:
        return pd.DataFrame(objects)
    else:
        return objects