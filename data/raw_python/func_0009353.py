def _unpack(c, tmp, package, version, git_url=None):
    """
    Download + unpack given package into temp dir ``tmp``.

    Return ``(real_version, source)`` where ``real_version`` is the "actual"
    version downloaded (e.g. if a Git master was indicated, it will be the SHA
    of master HEAD) and ``source`` is the source directory (relative to
    unpacked source) to import into ``<project>/vendor``.
    """
    real_version = version[:]
    source = None
    if git_url:
        pass
    #   git clone into tempdir
    #   git checkout <version>
    #   set target to checkout
    #   if version does not look SHA-ish:
    #       in the checkout, obtain SHA from that branch
    #       set real_version to that value
    else:
        cwd = os.getcwd()
        print("Moving into temp dir %s" % tmp)
        os.chdir(tmp)
        try:
            # Nab from index. Skip wheels; we want to unpack an sdist.
            flags = "--download=. --build=build --no-use-wheel"
            cmd = "pip install %s %s==%s" % (flags, package, version)
            c.run(cmd)
            # Identify basename
            # TODO: glob is bad here because pip install --download gets all
            # dependencies too! ugh. Figure out best approach for that.
            globs = []
            globexpr = ""
            for extension, opener in (
                ("zip", "unzip"),
                ("tgz", "tar xzvf"),
                ("tar.gz", "tar xzvf"),
            ):
                globexpr = "*.{0}".format(extension)
                globs = glob(globexpr)
                if globs:
                    break
            archive = os.path.basename(globs[0])
            source, _, _ = archive.rpartition(".{0}".format(extension))
            c.run("{0} {1}".format(opener, globexpr))
        finally:
            os.chdir(cwd)
    return real_version, source