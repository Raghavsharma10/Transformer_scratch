def distro_check():
    """Return a string containing the distro package manager."""
    distro_data = platform.linux_distribution()
    distro = [d.lower() for d in distro_data if d.isalpha()]

    if any(['ubuntu' in distro, 'debian' in distro]) is True:
        return 'apt'
    elif any(['centos' in distro, 'redhat' in distro]) is True:
        return 'yum'
    elif any(['suse' in distro]) is True:
        return 'zypper'
    else:
        raise AssertionError(
            'Distro [ %s ] is unsupported.' % distro
        )