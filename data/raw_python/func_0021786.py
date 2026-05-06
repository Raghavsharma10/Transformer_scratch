def get_metadata(dist):
    """
    Return dictionary of metadata for given dist

    @param dist: distribution
    @type dist: pkg_resources Distribution object

    @returns: dict of metadata or None

    """
    if not dist.has_metadata('PKG-INFO'):
        return

    msg = email.message_from_string(dist.get_metadata('PKG-INFO'))
    metadata = {}
    for header in [l for l in msg._headers]:
        metadata[header[0]] = header[1]

    return metadata