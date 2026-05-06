def run_license_checker(config_path):
    # type: (str) -> None
    """Generate table of installed packages and check for license
    warnings based off user defined restricted license values.

    :param config_path: str
    :return:
    """
    whitelist_licenses = _get_whitelist_licenses(config_path)
    table = PrintTable(ROW_HEADERS)

    warnings = []

    for pkg in _get_packages():
        allowed = pkg.license in whitelist_licenses
        table.add_row((pkg.name, pkg.version, pkg.license, str(allowed)))

        if not allowed:
            warnings.append(pkg)

    print(table)

    print('{} RESTRICTED LICENSES DETECTED'.format(len(warnings)))