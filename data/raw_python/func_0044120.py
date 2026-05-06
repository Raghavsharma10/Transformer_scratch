def load_files(files):
    """Load and execute a python file."""

    for py_file in files:
        LOG.debug("exec %s", py_file)
        execfile(py_file, globals(), locals())