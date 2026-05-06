def cwd():
    """Return the be current working directory"""
    cwd = os.environ.get("BE_CWD")
    if cwd and not os.path.isdir(cwd):
        sys.stderr.write("ERROR: %s is not a directory" % cwd)
        sys.exit(lib.USER_ERROR)
    return cwd or os.getcwd().replace("\\", "/")