def create_files(filedef, cleanup=True):
    """Contextmanager that creates a directory structure from a yaml
       descripttion.
    """
    cwd = os.getcwd()
    tmpdir = tempfile.mkdtemp()
    try:
        Filemaker(tmpdir, filedef)
        if not cleanup:  # pragma: nocover
            pass
            # print("TMPDIR =", tmpdir)
        os.chdir(tmpdir)
        yield tmpdir
    finally:
        os.chdir(cwd)
        if cleanup:  # pragma: nocover
            shutil.rmtree(tmpdir, ignore_errors=True)