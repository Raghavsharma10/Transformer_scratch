def restore(mongo_user, mongo_password, backup_tbz_path,
            backup_directory_output_path="/tmp/mongo_dump",
            drop_database=False, cleanup=True, silent=False,
            skip_system_and_user_files=False):
    """
    Runs mongorestore with source data from the provided .tbz backup, using
    the provided username and password.
    The contents of the .tbz will be dumped into the provided backup directory,
    and that folder will be deleted after a successful mongodb restore unless
    cleanup is set to False.
    
    Note: the skip_system_and_user_files is intended for use with the changes
    in user architecture introduced in mongodb version 2.6.
    
    Warning: Setting drop_database to True will drop the ENTIRE
    CURRENTLY RUNNING DATABASE before restoring.
    
    Mongorestore requires a running mongod process, in addition the provided
    user must have restore permissions for the database.  A mongolia superuser
    will have more than adequate permissions, but a regular user may not.
    By default this function will clean up the output of the untar operation.
    """
    if not path.exists(backup_tbz_path):
        raise Exception("the provided tar file %s does not exist." % (backup_tbz_path))
    
    untarbz(backup_tbz_path, backup_directory_output_path, silent=silent)
    
    if skip_system_and_user_files:
        system_and_users_path = "%s/admin" % backup_directory_output_path
        if path.exists(system_and_users_path):
            rmtree(system_and_users_path)
    
    mongorestore(mongo_user, mongo_password, backup_directory_output_path,
                 drop_database=drop_database, silent=silent)
    if cleanup:
        rmtree(backup_directory_output_path)