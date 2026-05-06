def _generate_uri(admin_metadata, base_uri):
    """Return dataset URI.

    :param admin_metadata: dataset administrative metadata
    :param base_uri: base URI from which to derive dataset URI
    :returns: dataset URI
    """
    name = admin_metadata["name"]
    uuid = admin_metadata["uuid"]
    # storage_broker_lookup = _generate_storage_broker_lookup()
    # parse_result = urlparse(base_uri)
    # storage = parse_result.scheme
    StorageBroker = _get_storage_broker(base_uri, config_path=None)
    return StorageBroker.generate_uri(name, uuid, base_uri)