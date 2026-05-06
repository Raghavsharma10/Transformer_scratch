def _get_storage_broker(uri, config_path):
    """Helper function to enable use lookup of appropriate storage brokers."""
    uri = dtoolcore.utils.sanitise_uri(uri)
    storage_broker_lookup = _generate_storage_broker_lookup()
    parsed_uri = dtoolcore.utils.generous_parse_uri(uri)
    StorageBroker = storage_broker_lookup[parsed_uri.scheme]
    return StorageBroker(uri, config_path)