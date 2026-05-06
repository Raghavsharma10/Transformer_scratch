def generate_delete_user_command(username=None, manage_home=None):
    """Generate command to delete a user.

    args:
        username (str): user name
        manage_home (bool): manage home directory

    returns:
        list: The user delete command string split into shell-like syntax
    """
    command = None
    remove_home = '-r' if manage_home else ''

    if get_platform() in ('Linux', 'OpenBSD'):
        command = '{0} {1} {2} {3}'.format(sudo_check(), LINUX_CMD_USERDEL, remove_home, username)
    elif get_platform() == 'FreeBSD':  # pragma: FreeBSD
        command = '{0} {1} userdel {2} -n {3}'.format(sudo_check(), FREEBSD_CMD_PW, remove_home, username)
    if command:
        return shlex.split(str(command))