def unzip(filename, match_dir=False, destdir=None):
    """
    Extract all files from a zip archive
    filename: The path to the zip file
    match_dir: If True all files in the zip must be contained in a subdirectory
      named after the archive file with extension removed
    destdir: Extract the zip into this directory, default current directory

    return: If match_dir is True then returns the subdirectory (including
      destdir), otherwise returns destdir or '.'
    """
    if not destdir:
        destdir = '.'

    z = zipfile.ZipFile(filename)
    unzipped = '.'

    if match_dir:
        if not filename.endswith('.zip'):
            raise FileException('Expected .zip file extension', filename)
        unzipped = os.path.basename(filename)[:-4]
        check_extracted_paths(z.namelist(), unzipped)
    else:
        check_extracted_paths(z.namelist())

    # File permissions, see
    # http://stackoverflow.com/a/6297838
    # http://stackoverflow.com/a/3015466
    for info in z.infolist():
        log.debug('Extracting %s to %s', info.filename, destdir)
        z.extract(info, destdir)
        perms = info.external_attr >> 16 & 4095
        if perms > 0:
            os.chmod(os.path.join(destdir, info.filename), perms)

    return os.path.join(destdir, unzipped)