def load_db(file, db, verbose=True):
    """
    Load :class:`mongomock.database.Database` from a local file.

    :param file: file path.
    :param db: instance of :class:`mongomock.database.Database`.
    :param verbose: bool, toggle on log.
    :return: loaded db.
    """
    db_data = json.load(file, verbose=verbose)
    return _load(db_data, db)