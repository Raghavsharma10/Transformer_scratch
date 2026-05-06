def list_database(db=None):
    """
    Lists the names of either the databases on the machine or the collections
    of a particular database
    
    @param db: the database for which to list the collection names;
        if db is None, then it lists all databases instead
        the contents of the database with the name passed in db
    """
    if db is None:
        return CONNECTION.get_connection().database_names()
    return CONNECTION.get_connection()[db].collection_names()