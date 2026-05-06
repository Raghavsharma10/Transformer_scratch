def mongorestore(mongo_user, mongo_password, backup_directory_path, drop_database=False, silent=False):
    """ Warning: Setting drop_database to True will drop the ENTIRE
        CURRENTLY RUNNING DATABASE before restoring.
        
        Mongorestore requires a running mongod process, in addition the provided
        user must have restore permissions for the database.  A mongolia superuser
        will have more than adequate permissions, but a regular user may not.
    """
    
    if not path.exists(backup_directory_path):
        raise Exception("the provided tar directory %s does not exist."
                        % (backup_directory_path))
    
    if silent:
        mongorestore_command = ("mongorestore --quiet -u %s -p %s %s"
                                % (mongo_user, mongo_password, backup_directory_path))
    else:
        mongorestore_command = ("mongorestore -v -u %s -p %s %s"
                                % (mongo_user, mongo_password, backup_directory_path))
    if drop_database:
        mongorestore_command = mongorestore_command + " --drop"
    call(mongorestore_command, silent=silent)