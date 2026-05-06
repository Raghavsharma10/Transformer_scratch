def _create_cached_db(
        db_path,
        tables,
        version=1):
    """
    Either create or retrieve sqlite database.

    Parameters
    --------
    db_path : str
        Path to sqlite3 database file

    tables : dict
        Dictionary mapping table names to datacache.DatabaseTable objects

    version : int, optional
        Version acceptable as cached data.

    Returns sqlite3 connection
    """
    require_string(db_path, "db_path")
    require_iterable_of(tables, DatabaseTable)
    require_integer(version, "version")

    # if the database file doesn't already exist and we encounter an error
    # later, delete the file before raising an exception
    delete_on_error = not exists(db_path)

    # if the database already exists, contains all the table
    # names and has the right version, then just return it
    db = Database(db_path)

    # make sure to delete the database file in case anything goes wrong
    # to avoid leaving behind an empty DB
    table_names = [table.name for table in tables]
    try:
        if db.has_tables(table_names) and \
                db.has_version() and \
                db.version() == version:
            logger.info("Found existing table in database %s", db_path)
        else:
            if len(db.table_names()) > 0:
                logger.info(
                    "Dropping tables from database %s: %s",
                    db_path,
                    ", ".join(db.table_names()))
                db.drop_all_tables()
            logger.info(
                "Creating database %s containing: %s",
                db_path,
                ", ".join(table_names))
            db.create(tables, version)
    except:
        logger.warning(
            "Failed to create tables %s in database %s",
            table_names,
            db_path)
        db.close()
        if delete_on_error:
            remove(db_path)
        raise
    return db.connection