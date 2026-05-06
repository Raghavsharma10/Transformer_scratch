def get_config_file(basename):
    """ Looks for a configuration file in 3 locations:

        - the current directory
        - the user config directory (~/.config/scriptabit)
        - the version installed with the package (using setuptools resource API)

    Args:
        basename (str): The base filename.

    Returns:
        str: The full path to the configuration file.
    """
    locations = [
        os.path.join(os.curdir, basename),
        os.path.join(
            os.path.expanduser("~"),
            ".config",
            "scriptabit",
            basename),
        resource_filename(
            Requirement.parse("scriptabit"),
            os.path.join('scriptabit', basename))
    ]

    for location in locations:
        if os.path.isfile(location):
            return location