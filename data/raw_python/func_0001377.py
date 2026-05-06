def generate_modify_user_command(task=None, manage_home=None):
    """Generate command to modify existing user to become the proposed user.

    args:
        task (dict): A proposed user and the differences between it and the existing user

    returns:
        list: The command string split into shell-like syntax
    """
    name = task['proposed_user'].name
    comparison_result = task['user_comparison']['result']
    command = None
    if get_platform() in ('Linux', 'OpenBSD'):
        command = '{0} {1}'.format(sudo_check(), LINUX_CMD_USERMOD)
        if comparison_result.get('replacement_uid_value'):
            command = '{0} -u {1}'.format(command, comparison_result.get('replacement_uid_value'))
        if comparison_result.get('replacement_gid_value'):
            command = '{0} -g {1}'.format(command, comparison_result.get('replacement_gid_value'))
        if comparison_result.get('replacement_gecos_value'):
            command = '{0} -c {1}'.format(command, comparison_result.get('replacement_gecos_value'))
        if comparison_result.get('replacement_shell_value'):
            command = '{0} -s {1}'.format(command, comparison_result.get('replacement_shell_value'))
        if manage_home and comparison_result.get('replacement_home_dir_value'):
                command = '{0} -d {1}'.format(command, comparison_result.get('replacement_home_dir_value'))
        command = '{0} {1}'.format(command, name)
    if get_platform() == 'FreeBSD':  # pragma: FreeBSD
        command = '{0} {1} usermod'.format(sudo_check(), FREEBSD_CMD_PW)
        if comparison_result.get('replacement_uid_value'):
            command = '{0} -u {1}'.format(command, comparison_result.get('replacement_uid_value'))
        if comparison_result.get('replacement_gid_value'):
            command = '{0} -g {1}'.format(command, comparison_result.get('replacement_gid_value'))
        if comparison_result.get('replacement_gecos_value'):
            command = '{0} -c {1}'.format(command, comparison_result.get('replacement_gecos_value'))
        if comparison_result.get('replacement_shell_value'):
            command = '{0} -s {1}'.format(command, comparison_result.get('replacement_shell_value'))
        if manage_home and comparison_result.get('replacement_home_dir_value'):
            command = '{0} -d {1}'.format(command, comparison_result.get('replacement_home_dir_value'))
        command = '{0} -n {1}'.format(command, name)
    if command:
        return shlex.split(str(command))