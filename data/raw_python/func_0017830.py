def delete_dataset_cache(*filenames):
    """
    Delete the cache (converted files) for a dataset.

    Parameters
    ----------
    filenames: str
        Filenames of files to delete
    """
    for filename in filenames:
        filename = path_string(filename)
        path = config.get_data_path(filename)
        if os.path.exists(path):
            os.remove(path)