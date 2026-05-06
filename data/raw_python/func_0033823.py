def get_version():
    """Reads version number.

    This workaround is required since __init__ is an entry point exposing
    stuff from other modules, which may use dependencies unavailable
    in current environment, which in turn will prevent this application
    from install.

    """
    contents = read(os.path.join(PATH_BASE, 'srptools', '__init__.py'))
    version = re.search('VERSION = \(([^)]+)\)', contents)
    version = version.group(1).replace(', ', '.').strip()
    return version