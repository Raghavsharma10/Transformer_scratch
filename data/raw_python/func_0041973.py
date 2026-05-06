def dump_db(db, file,
            pretty=False,
            overwrite=False,
            verbose=True):
    """
    Dump :class:`mongomock.database.Database` to a local file. Only support
    ``*.json`` or ``*.gz`` (compressed json file)

    :param db: instance of :class:`mongomock.database.Database`.
    :param file: file path.
    :param pretty: bool, toggle on jsonize into pretty format.
    :param overwrite: bool, allow overwrite to existing file.
    :param verbose: bool, toggle on log.
    """
    db_data = _dump(db)
    json.dump(
        db_data, file,
        pretty=pretty, overwrite=overwrite, verbose=verbose,
    )