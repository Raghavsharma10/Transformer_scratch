def warn_if_outdated(package,
                     version,
                     raise_exceptions=False,
                     background=True,
                     ):
    """
    Higher level convenience function using check_outdated.

    The package and version arguments are the same.

    If the package is outdated, a warning (OutdatedPackageWarning) will
    be emitted.

    Any exception in check_outdated will be converted to a warning (OutdatedCheckFailedWarning)
    unless raise_exceptions if True.

    If background is True (the default), the check will run in
    a background thread so this function will return immediately.
    In this case if an exception is raised and raise_exceptions if True
    the traceback will be printed to stderr but the program will not be
    interrupted.

    This function doesn't return anything.
    """

    def check():
        # noinspection PyUnusedLocal
        is_outdated = False
        with utils.exception_to_warning('check for latest version of package',
                                        OutdatedCheckFailedWarning,
                                        always_raise=raise_exceptions):
            is_outdated, latest = check_outdated(package, version)

        if is_outdated:
            warn_with_ignore(
                'The package %s is out of date. Your version is %s, the latest is %s.'
                % (package, version, latest),
                OutdatedPackageWarning,
            )

    if background:
        thread = Thread(target=check)
        thread.start()
    else:
        check()