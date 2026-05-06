def get_version(dev_version=False):
    """Generates a version string.

    Arguments:
        dev_version: Generate a verbose development version from git commits.

    Examples:
        1.1
        1.1.dev43 # If 'dev_version' was passed.
    """
    if dev_version:
        version = git_dev_version()
        if not version:
            raise RuntimeError("Could not generate dev version from git.")

        return version

    return "1!%d.%d" % (MAJOR, MINOR)