def get_as_local_path(path, overwrite, progress=0,
                      httpuser=None, httppassword=None):
    """
    Automatically handle local and remote URLs, files and directories

    path: Either a local directory, file or remote URL. If a URL is given
      it will be fetched. If this is a zip it will be automatically
      expanded by default.
    overwrite: Whether to overwrite an existing file:
      'error': Raise an exception
      'backup: Renamed the old file and use the new one
      'keep': Keep the old file, don't overwrite or raise an exception
    progress: Number of progress dots, default 0 (don't print)
    httpuser, httppass: Credentials for HTTP authentication
    return: A tuple (type, localpath)
      type:
        'file': localpath is the path to a local file
        'directory': localpath is the path to a local directory
        'unzipped': localpath is the path to a local unzipped directory
    """
    m = re.match('([A-Za-z]+)://', path)
    if m:
        # url_open handles multiple protocols so don't bother validating
        log.debug('Detected URL protocol: %s', m.group(1))

        # URL should use / as the pathsep
        localpath = path.split('/')[-1]
        if not localpath:
            raise FileException(
                'Remote path appears to be a directory', path)

        if os.path.exists(localpath):
            if overwrite == 'error':
                raise FileException('File already exists', localpath)
            elif overwrite == 'keep':
                log.info('Keeping existing %s', localpath)
            elif overwrite == 'backup':
                rename_backup(localpath)
                download(path, localpath, progress, httpuser=httpuser,
                         httppassword=httppassword)
            else:
                raise Exception('Invalid overwrite flag: %s' % overwrite)
        else:
            download(path, localpath, progress, httpuser=httpuser,
                     httppassword=httppassword)
    else:
        localpath = path
    log.debug("Local path: %s", localpath)

    if os.path.isdir(localpath):
        return 'directory', localpath
    if os.path.exists(localpath):
        return 'file', localpath

    # Somethings gone very wrong
    raise Exception('Local path does not exist: %s' % localpath)