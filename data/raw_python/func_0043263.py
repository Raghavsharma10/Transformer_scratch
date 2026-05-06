def _create_conda_cmd(conda_cmd, args=None, env=None, user=None):
    """
    Utility to create a valid conda command
    """
    cmd = [_get_conda_path(user=user), conda_cmd]
    if env:
        cmd.extend(['-n', env])
    if args is not None and isinstance(args, list) and args != []:
        cmd.extend(args)
    return cmd