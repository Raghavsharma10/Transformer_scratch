def configure(root_directory, build_path, cmake_command, only_show):
    """
    Main configure function.
    """
    default_build_path = os.path.join(root_directory, 'build')

    # check that CMake is available, if not stop
    check_cmake_exists('cmake')

    # deal with build path
    if build_path is None:
        build_path = default_build_path
    if not only_show:
        setup_build_path(build_path)

    cmake_command += ' -B' + build_path
    print('{0}\n'.format(cmake_command))
    if only_show:
        sys.exit(0)

    run_cmake(cmake_command, build_path, default_build_path)