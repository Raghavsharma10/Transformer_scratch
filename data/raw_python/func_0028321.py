def _make_cmd_list(cmd_list):
    """
    Helper function to easily create the proper json formated string from a list of strs
    :param cmd_list: list of strings
    :return: str json formatted
    """
    cmd = ''
    for i in cmd_list:
        cmd = cmd + '"' + i + '",'
    cmd = cmd[:-1]
    return cmd