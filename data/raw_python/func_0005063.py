def pylint_check(files):
    # type: (List[str]) -> int
    """ Run code checks using pylint.

    Args:
        files (list[str]):
            A list of files to check

    Returns:
        bool: **True** if all files passed the checks, **False** otherwise.
    """
    files = fs.wrap_paths(files)
    cfg_path = conf.get_path('lint.pylint_cfg', 'ops/tools/pylint.ini')
    pylint_cmd = 'pylint --rcfile {} {}'.format(cfg_path, files)

    return shell.run(pylint_cmd, exit_on_error=False).return_code