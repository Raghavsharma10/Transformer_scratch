def pipinstall(packages):
    """
    Install one or more pip packages.

    :type packages: string or list
    :param packages: The package or list of packages to install.

    :raises TypeError: Nor a string or a list was provided.
    """

    if isinstance(packages, str):
        if hasattr(pip, 'main'):
            pip.main(['install', packages])
        else:
            pip._internal.main(['install', packages])
    elif isinstance(packages, list):
        for i in enumerate(packages):
            if hasattr(pip, 'main'):
                pip.main(['install', i[1]])
            else:
                pip._internal.main(['install', i[1]])
    else:
        raise TypeError("Nor a string or a list was provided.")