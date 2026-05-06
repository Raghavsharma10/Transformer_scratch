def remove_container(name, force=False):
    """
    Wrapper for docker remove_container

    :returns: True if container was found and removed
    """

    try:
        if not force:
            _get_docker().stop(name)
    except APIError:
        pass
    try:
        _get_docker().remove_container(name, force=True)
        return True
    except APIError:
        return False