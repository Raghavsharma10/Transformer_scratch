def run(targets, config_dir='.', check_licenses=False):
    # type: (List[str], str, bool) -> None
    """Runs `pylint` and `flake8` commands and exits based off the evaluation
    of both command results.

    :param targets: List[str]
    :param config_dir: str
    :param check_licenses: bool
    :return:
    """
    pylint_return_state = False
    flake8_return_state = False

    if check_licenses:
        run_license_checker(config_path=get_license_checker_config_path(config_dir))

    pylint_options = get_pylint_options(config_dir=config_dir)
    flake8_options = get_flake8_options(config_dir=config_dir)

    if targets:
        pylint_return_state = _run_command(command='pylint', targets=targets,
                                           options=pylint_options)
        flake8_return_state = _run_command(command='flake8', targets=targets,
                                           options=flake8_options)

    if not flake8_return_state and not pylint_return_state:
        sys.exit(0)
    else:
        sys.exit(1)