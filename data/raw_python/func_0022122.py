def _version_string():
    """ Gets the output for `trytravis --version`. """
    platform_system = platform.system()
    if platform_system == 'Linux':
        os_name, os_version, _ = platform.dist()
    else:
        os_name = platform_system
        os_version = platform.version()
    python_version = platform.python_version()
    return 'trytravis %s (%s %s, python %s)' % (__version__,
                                                os_name.lower(),
                                                os_version,
                                                python_version)