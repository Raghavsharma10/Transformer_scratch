def authenticate_connection(username, password, db=None):
    """
    Authenticates the current database connection with the passed username
    and password.  If the database connection uses all default parameters,
    this can be called without connect_to_database.  Otherwise, it should
    be preceded by a connect_to_database call.
    
    @param username: the username with which you authenticate; must match
        a user registered in the database
    @param password: the password of that user
    @param db: the database the user is authenticated to access.  Passing None
    (the default) means authenticating against the admin database, which
    gives the connection access to all databases
    
    Example; connecting to all databases locally:
        connect_to_database()
        authenticate_connection("username", "password")
    
    Example; connecting to a particular database of a remote server:
        connect_to_database(host="example.com", port="12345")
        authenticate_connection("username", "password", db="somedb")
    
    """
    return CONNECTION.authenticate(username, password, db=db)