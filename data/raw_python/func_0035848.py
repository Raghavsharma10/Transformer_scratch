def _admin_metadata_from_uri(uri, config_path):
    """Helper function for getting admin metadata."""
    uri = dtoolcore.utils.sanitise_uri(uri)
    storage_broker = _get_storage_broker(uri, config_path)
    admin_metadata = storage_broker.get_admin_metadata()
    return admin_metadata