def _find_package(c):
    """
    Try to find 'the' One True Package for this project.

    Mostly for obtaining the ``_version`` file within it.

    Uses the ``packaging.package`` config setting if defined. If not defined,
    fallback is to look for a single top-level Python package (directory
    containing ``__init__.py``). (This search ignores a small blacklist of
    directories like ``tests/``, ``vendor/`` etc.)
    """
    # TODO: is there a way to get this from the same place setup.py does w/o
    # setup.py barfing (since setup() runs at import time and assumes CLI use)?
    configured_value = c.get("packaging", {}).get("package", None)
    if configured_value:
        return configured_value
    # TODO: tests covering this stuff here (most logic tests simply supply
    # config above)
    packages = [
        path
        for path in os.listdir(".")
        if (
            os.path.isdir(path)
            and os.path.exists(os.path.join(path, "__init__.py"))
            and path not in ("tests", "integration", "sites", "vendor")
        )
    ]
    if not packages:
        sys.exit("Unable to find a local Python package!")
    if len(packages) > 1:
        sys.exit("Found multiple Python packages: {0!r}".format(packages))
    return packages[0]