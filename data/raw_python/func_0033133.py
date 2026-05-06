def add_user(name, password=None, read_only=None, db=None, **kwargs):
    """
    Adds a user that can be used for authentication.
    
    @param name: the name of the user to create
    @param passowrd: the password of the user to create. Can not be used with
        the userSource argument.
    @param read_only: if True the user will be read only
    @param db: the database the user is authenticated to access.  Passing None
        (the default) means add the user to the admin database, which gives the
        user access to all databases
    @param **kwargs: forwarded to pymongo.database.add_user
    
    Example; adding a user with full database access:
        add_user("username", "password")
    
    Example; adding a user with read only privilage on a partiucalr database:
        add_user("username", "password", read_only=True, db="somedb")
    
    NOTE: This function will only work if mongo is being run unauthenticated
    or you have already authenticated with another user with appropriate
    privileges to add a user to the specified database.
    """
    return CONNECTION.add_user(name, password=password, read_only=read_only, db=db, **kwargs)