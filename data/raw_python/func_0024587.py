def download_data(cdbs_root, verbose=True, dry_run=False):
    """Download CDBS data files to given root directory.
    Download is skipped if a data file already exists.

    Parameters
    ----------
    cdbs_root : str
        Root directory for CDBS data files.

    verbose : bool
        Print extra information to screen.

    dry_run : bool
        Go through the logic but skip the actual download.
        This would return a list of files that *would have been*
        downloaded without network calls.
        Use this option for debugging or testing.

    Raises
    ------
    OSError
        Problem with directory.

    Returns
    -------
    file_list : list of str
        A list of downloaded files.

    """
    from .config import conf  # Avoid potential circular import

    if not os.path.exists(cdbs_root):
        os.makedirs(cdbs_root, exist_ok=True)
        if verbose:  # pragma: no cover
            print('Created {}'.format(cdbs_root))
    elif not os.path.isdir(cdbs_root):
        raise OSError('{} must be a directory'.format(cdbs_root))

    host = 'http://ssb.stsci.edu/cdbs/'
    file_list = []

    if not cdbs_root.endswith(os.sep):
        cdbs_root += os.sep

    # See https://github.com/astropy/astropy/issues/8524
    for cfgitem in conf.__class__.__dict__.values():
        if (not isinstance(cfgitem, ConfigItem) or
                not cfgitem.name.endswith('file')):
            continue

        url = cfgitem()

        if not url.startswith(host):
            if verbose:  # pragma: no cover
                print('{} is not from {}, skipping download'.format(
                    url, host))
            continue

        dst = url.replace(host, cdbs_root).replace('/', os.sep)

        if os.path.exists(dst):
            if verbose:  # pragma: no cover
                print('{} already exists, skipping download'.format(dst))
            continue

        # Create sub-directories, if needed.
        subdirs = os.path.dirname(dst)
        os.makedirs(subdirs, exist_ok=True)

        if not dry_run:  # pragma: no cover
            try:
                src = download_file(url)
                copyfile(src, dst)
            except Exception as exc:
                print('Download failed - {}'.format(str(exc)))
                continue

        file_list.append(dst)
        if verbose:  # pragma: no cover
            print('{} downloaded to {}'.format(url, dst))

    return file_list