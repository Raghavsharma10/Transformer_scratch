def get(datasets_identifiers, identifier_type='hid', history_id=None):
    """
        Given the history_id that is displayed to the user, this function will
        download the file[s] from the history and stores them under /import/
        Return value[s] are the path[s] to the dataset[s] stored under /import/
    """
    history_id = history_id or os.environ['HISTORY_ID']
    # The object version of bioblend is to slow in retrieving all datasets from a history
    # fallback to the non-object path
    gi = get_galaxy_connection(history_id=history_id, obj=False)
    for dataset_identifier in datasets_identifiers:
        file_path = '/import/%s' % dataset_identifier
        log.debug('Downloading gx=%s history=%s dataset=%s', gi, history_id, dataset_identifier)
        # Cache the file requests. E.g. in the example of someone doing something
        # silly like a get() for a Galaxy file in a for-loop, wouldn't want to
        # re-download every time and add that overhead.
        if not os.path.exists(file_path):
            hc = HistoryClient(gi)
            dc = DatasetClient(gi)
            history = hc.show_history(history_id, contents=True)
            datasets = {ds[identifier_type]: ds['id'] for ds in history}
            if identifier_type == 'hid':
                dataset_identifier = int(dataset_identifier)
            dc.download_dataset(datasets[dataset_identifier], file_path=file_path, use_default_filename=False)
        else:
            log.debug('Cached, not re-downloading')

    return file_path