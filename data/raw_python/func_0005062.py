def pep8_check(files):
    # type: (List[str]) -> int
    """ Run code checks using pep8.

    Args:
        files (list[str]):
            A list of files to check

    Returns:
        bool: **True** if all files passed the checks, **False** otherwise.

    pep8 tool is **very** fast. Especially compared to pylint and the bigger the
    code base the bigger the difference. If you want to reduce check times you
    might disable all pep8 checks in pylint and use pep8 for that. This way you
    use pylint only for the more advanced checks (the number of checks enabled
    in pylint will make a visible difference in it's run times).
    """
    files = fs.wrap_paths(files)
    cfg_path = conf.get_path('lint.pep8_cfg', 'ops/tools/pep8.ini')
    pep8_cmd = 'pep8 --config {} {}'.format(cfg_path, files)

    return shell.run(pep8_cmd, exit_on_error=False).return_code