def prepare_storage_dir(storage_directory):
    """Prepare the storage directory."""
    storage_directory = os.path.expanduser(storage_directory)
    if not os.path.exists(storage_directory):
        os.mkdir(storage_directory)

    return storage_directory