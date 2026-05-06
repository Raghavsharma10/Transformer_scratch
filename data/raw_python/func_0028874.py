def operating_system():
    """Return a string identifying the operating system the application
    is running on.

    :rtype: str

    """
    if platform.system() == 'Darwin':
        return 'OS X Version %s' % platform.mac_ver()[0]
    distribution = ' '.join(platform.linux_distribution()).strip()
    os_platform = platform.platform(True, True)
    if distribution:
        os_platform += ' (%s)' % distribution
    return os_platform