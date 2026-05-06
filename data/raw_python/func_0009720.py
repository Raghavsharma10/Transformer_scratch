def show_tables():
    """
    Return the names of the tables currently in the database.
    """
    _State.connection()
    _State.reflect_metadata()
    metadata = _State.metadata

    response = select('name, sql from sqlite_master where type="table"')

    return {row['name']: row['sql'] for row in response}