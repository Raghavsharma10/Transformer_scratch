def zip(filename, paths, strip_prefix=''):
    """
    Create a new zip archive containing files
    filename: The name of the zip file to be created
    paths: A list of files or directories
    strip_dir: Remove this prefix from all file-paths before adding to zip
    """
    if isinstance(paths, basestring):
        paths = [paths]

    filelist = set()
    for p in paths:
        if os.path.isfile(p):
            filelist.add(p)
        else:
            for root, dirs, files in os.walk(p):
                for f in files:
                    filelist.add(os.path.join(root, f))

    z = zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED)
    for f in sorted(filelist):
        arcname = f
        if arcname.startswith(strip_prefix):
            arcname = arcname[len(strip_prefix):]
        if arcname.startswith(os.path.sep):
            arcname = arcname[1:]
        log.debug('Adding %s to %s[%s]', f, filename, arcname)
        z.write(f, arcname)

    z.close()