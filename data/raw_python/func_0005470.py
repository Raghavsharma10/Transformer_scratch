def is_systemd():
    """Determines whether this system uses systemd

    :return: (bool) True if this distro has systemd
    """
    os_family = platform.system()
    if os_family != 'Linux':
        raise OSError('This method is only supported on Linux, found OS: {o}'.format(o=os_family))
    linux_distro, linux_version, distro_name = platform.linux_distribution()

    # Determine when to use systemd
    systemd = False
    if 'ubuntu' in linux_distro.lower() and '16' in linux_version:
        systemd = True
    elif 'red' in linux_distro.lower() and '7' in linux_version:
        systemd = True
    elif 'cent' in linux_distro.lower() and '7' in linux_version:
        systemd = True
    return systemd