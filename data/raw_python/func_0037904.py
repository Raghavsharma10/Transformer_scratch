def setup_build_path(build_path):
    """
    Create build directory. If this already exists, print informative
    error message and quit.
    """
    if os.path.isdir(build_path):
        fname = os.path.join(build_path, 'CMakeCache.txt')
        if os.path.exists(fname):
            sys.stderr.write('aborting setup\n')
            sys.stderr.write(
                'build directory {0} which contains CMakeCache.txt already exists\n'.
                format(build_path))
            sys.stderr.write(
                'remove the build directory and then rerun setup\n')
            sys.exit(1)
    else:
        os.makedirs(build_path, 0o755)