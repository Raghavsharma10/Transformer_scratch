def _validate_charm(url, service_name, add_error):
    """Validate the given charm URL.

    Use the given service name to describe possible errors.
    Use the given add_error callable to register validation error.

    If the URL is valid, return the corresponding charm reference object.
    Return None otherwise.
    """
    if url is None:
        add_error('no charm specified for service {}'.format(service_name))
        return None
    if not isstring(url):
        add_error(
            'invalid charm specified for service {}: {}'
            ''.format(service_name, url))
        return None
    if not url.strip():
        add_error('empty charm specified for service {}'.format(service_name))
        return None
    try:
        charm = references.Reference.from_string(url)
    except ValueError as e:
        msg = pyutils.exception_string(e)
        add_error(
            'invalid charm specified for service {}: {}'
            ''.format(service_name, msg))
        return None
    if charm.is_local():
        add_error(
            'local charms not allowed for service {}: {}'
            ''.format(service_name, charm))
        return None
    if charm.is_bundle():
        add_error(
            'bundle cannot be used as charm for service {}: {}'
            ''.format(service_name, charm))
        return None
    return charm