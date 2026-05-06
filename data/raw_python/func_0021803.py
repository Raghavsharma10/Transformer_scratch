def mongodump(mongo_user, mongo_password, mongo_dump_directory_path, database=None, silent=False):
    """ Runs mongodump using the provided credentials on the running mongod
        process.
        
        WARNING: This function will delete the contents of the provided
        directory before it runs. """
    if path.exists(mongo_dump_directory_path):
        # If a backup dump already exists, delete it
        rmtree(mongo_dump_directory_path)
    if silent:
        dump_command = ("mongodump --quiet -u %s -p %s -o %s"
                        % (mongo_user, mongo_password, mongo_dump_directory_path))
    else:
        dump_command = ("mongodump -u %s -p %s -o %s"
                        % (mongo_user, mongo_password, mongo_dump_directory_path))
    if database:
        dump_command += (" --db %s" % database)
    call(dump_command, silent=silent)