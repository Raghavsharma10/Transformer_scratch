def generate_add_user_command(proposed_user=None, manage_home=None):
    """Generate command to add a user.

    args:
        proposed_user (User): User
        manage_home: bool

    returns:
        list: The command string split into shell-like syntax
    """
    command = None
    if get_platform() in ('Linux', 'OpenBSD'):
        command = '{0} {1}'.format(sudo_check(), LINUX_CMD_USERADD)
        if proposed_user.uid:
            command = '{0} -u {1}'.format(command, proposed_user.uid)
        if proposed_user.gid:
            command = '{0} -g {1}'.format(command, proposed_user.gid)
        if proposed_user.gecos:
            command = '{0} -c \'{1}\''.format(command, proposed_user.gecos)
        if manage_home:
            if proposed_user.home_dir:
                if os.path.exists(proposed_user.home_dir):
                    command = '{0} -d {1}'.format(command, proposed_user.home_dir)
            elif not os.path.exists('/home/{0}'.format(proposed_user.name)):
                command = '{0} -m'.format(command)
        if proposed_user.shell:
            command = '{0} -s {1}'.format(command, proposed_user.shell)
        command = '{0} {1}'.format(command, proposed_user.name)
    elif get_platform() == 'FreeBSD':  # pragma: FreeBSD
        command = '{0} {1} useradd'.format(sudo_check(), FREEBSD_CMD_PW)
        if proposed_user.uid:
            command = '{0} -u {1}'.format(command, proposed_user.uid)
        if proposed_user.gid:
            command = '{0} -g {1}'.format(command, proposed_user.gid)
        if proposed_user.gecos:
            command = '{0} -c \'{1}\''.format(command, proposed_user.gecos)
        if manage_home:
            if proposed_user.home_dir:
                command = '{0} -d {1}'.format(command, proposed_user.home_dir)
            else:
                command = '{0} -m'.format(command)
        if proposed_user.shell:
            command = '{0} -s {1}'.format(command, proposed_user.shell)
        command = '{0} -n {1}'.format(command, proposed_user.name)

    if command:
        return shlex.split(str(command))