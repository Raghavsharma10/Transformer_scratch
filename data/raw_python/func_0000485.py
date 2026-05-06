def put(filenames, file_type='auto', history_id=None):
    """
        Given filename[s] of any file accessible to the docker instance, this
        function will upload that file[s] to galaxy using the current history.
        Does not return anything.
    """
    history_id = history_id or os.environ['HISTORY_ID']
    gi = get_galaxy_connection(history_id=history_id)
    for filename in filenames:
        log.debug('Uploading gx=%s history=%s localpath=%s ft=%s', gi, history_id, filename, file_type)
        history = gi.histories.get(history_id)
        history.upload_dataset(filename, file_type=file_type)