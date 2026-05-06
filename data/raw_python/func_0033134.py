def add_superuser(name, password, **kwargs):
    """
    Adds a user with userAdminAnyDatabase role to mongo.
    
    @param name: the name of the user to create
    @param passowrd: the password of the user to create. Can not be used with
        the userSource argument.
    @param **kwargs: forwarded to pymongo.database.add_user
    """
    return CONNECTION.add_user(
            name, password=password,
            roles=["userAdminAnyDatabase", "readWriteAnyDatabase", "root", "backup", "restore"], **kwargs
    )