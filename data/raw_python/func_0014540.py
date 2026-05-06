def make_tarball(base_name, base_dir, compress='gzip',
                 verbose=False, dry_run=False):
    """Create a tar file from all the files under 'base_dir'.
    This file may be compressed.

    :param compress: Compression algorithms. Supported algorithms are:
        'gzip': (the default)
        'compress'
        'bzip2'
        None
    For 'gzip' and 'bzip2' the internal tarfile module will be used.
    For 'compress' the .tar will be created using tarfile, and then
    we will spawn 'compress' afterwards.
    The output tar file will be named 'base_name' + ".tar", 
    possibly plus the appropriate compression extension (".gz",
    ".bz2" or ".Z").  Return the output filename.
    """
    # XXX GNU tar 1.13 has a nifty option to add a prefix directory.
    # It's pretty new, though, so we certainly can't require it --
    # but it would be nice to take advantage of it to skip the
    # "create a tree of hardlinks" step!  (Would also be nice to
    # detect GNU tar to use its 'z' option and save a step.)

    compress_ext = { 'gzip': ".gz",
                     'bzip2': '.bz2',
                     'compress': ".Z" }

    # flags for compression program, each element of list will be an argument
    tarfile_compress_flag = {'gzip':'gz', 'bzip2':'bz2'}
    compress_flags = {'compress': ["-f"]}

    if compress is not None and compress not in compress_ext.keys():
        raise ValueError("bad value for 'compress': must be None, 'gzip',"
                         "'bzip2' or 'compress'")

    archive_name = base_name + ".tar"
    if compress and compress in tarfile_compress_flag:
        archive_name += compress_ext[compress]

    mode = 'w:' + tarfile_compress_flag.get(compress, '')

    mkpath(os.path.dirname(archive_name), dry_run=dry_run)
    log.info('Creating tar file %s with mode %s' % (archive_name, mode))

    if not dry_run:
        tar = tarfile.open(archive_name, mode=mode)
        # This recursively adds everything underneath base_dir
        tar.add(base_dir)
        tar.close()

    if compress and compress not in tarfile_compress_flag:
        spawn([compress] + compress_flags[compress] + [archive_name],
              dry_run=dry_run)
        return archive_name + compress_ext[compress]
    else:
        return archive_name