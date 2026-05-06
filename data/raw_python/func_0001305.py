def get_missing_commands(_platform):
    """Check I can identify the necessary commands for managing users."""
    missing = list()
    if _platform in ('Linux', 'OpenBSD'):
        if not LINUX_CMD_USERADD:
            missing.append('useradd')
        if not LINUX_CMD_USERMOD:
            missing.append('usermod')
        if not LINUX_CMD_USERDEL:
            missing.append('userdel')
        if not LINUX_CMD_GROUP_ADD:
            missing.append('groupadd')
        if not LINUX_CMD_GROUP_DEL:
            missing.append('groupdel')
    elif _platform == 'FreeBSD':  # pragma: FreeBSD
        # FREEBSD COMMANDS
        if not FREEBSD_CMD_PW:
            missing.append('pw')
    if missing:
        print('\nMISSING = {0}'.format(missing))
    return missing