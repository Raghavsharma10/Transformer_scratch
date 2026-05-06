def _is_dataset(uri, config_path):
    """Helper function for determining if a URI is a dataset."""
    uri = dtoolcore.utils.sanitise_uri(uri)
    storage_broker = _get_storage_broker(uri, config_path)
    return storage_broker.has_admin_metadata()