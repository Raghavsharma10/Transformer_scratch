def install_vexts(vext_files, verify=True):
    """
    copy vext_file to sys.prefix + '/share/vext/specs'

    (PIP7 seems to remove data_files so we recreate something similar here)
    """
    if verify and not check_sysdeps(vext_files):
        return

    spec_dir = join(prefix, 'share/vext/specs')
    try:
        makedirs(spec_dir)
    except OSError as e:
        if not isdir(spec_dir):
            logger.error("Error making spec directory [%s]: %r" % (spec_dir, e))

    for vext_file in vext_files:
        dest = normpath(join(spec_dir, basename(vext_file)))
        try:
            logger.debug("%s > %s" % (vext_file, dest))
            copyfile(vext_file, dest)
            yield vext_file, dest
        except IOError as e:
            logger.error("Could not copy %s %r" % (vext_file, e))