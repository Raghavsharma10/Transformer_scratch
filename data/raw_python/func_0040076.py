def requirements(requirements_file):
    """Return packages mentioned in the given file.

    Args:
        requirements_file (str): path to the requirements file to be parsed.

    Returns:
        (list): 3rd-party package dependencies contained in the file.
    """
    return [
        str(pkg.req) for pkg in parse_requirements(
            requirements_file, session=pip_download.PipSession()) if pkg.req is not None]