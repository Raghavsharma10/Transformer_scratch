def create_database(dbpath, schema='', overwrite=True):
    """
    Create a new database at the given dbpath

    Parameters
    ----------
    dbpath: str
        The full path for the new database, including the filename and .db file extension.
    schema: str
        The path to the .sql schema for the database
    overwrite: bool
        Overwrite dbpath if it already exists
    """
    if dbpath.endswith('.db'):
        
        if os.path.isfile(dbpath) and overwrite:
            os.system('rm {}'.format(dbpath))
        
        # Load the schema if given
        if schema:
            os.system("cat {} | sqlite3 {}".format(schema,dbpath))
            
        # Otherwise just make an empty SOURCES table
        else:
            sources_table = "CREATE TABLE sources (id INTEGER PRIMARY KEY, ra REAL, dec REAL, designation TEXT, " \
                            "publication_id INTEGER, shortname TEXT, names TEXT, comments TEXT)"
            os.system("sqlite3 {} '{}'".format(dbpath, sources_table))
            
        if os.path.isfile(dbpath):
            print(
                "\nDatabase created! To load, run\n\ndb = astrodb.Database('{}')"
                "\n\nThen run db.modify_table() method to create tables.".format(dbpath))
    else:
        print("Please provide a path and file name with a .db file extension, e.g. /Users/<username>/Desktop/test.db")