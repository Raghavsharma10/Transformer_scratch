def db(cls, path=None):
        """
        Returns a pymongo Collection object from the current database connection.
        If the database connection is in test mode, collection will be in the
        test database.
        
        @param path: if is None, the PATH attribute of the current class is used;
            if is not None, this is used instead
        
        @raise Exception: if neither cls.PATH or path are valid
        """
        if cls.PATH is None and path is None:
            raise Exception("No database specified")
        if path is None:
            path = cls.PATH
        if "." not in path:
            raise Exception(('invalid path "%s"; database paths must be ' +
                             'of the form "database.collection"') % (path,))
        if CONNECTION.test_mode:
            return CONNECTION.get_connection()[TEST_DATABASE_NAME][path]
        (db, coll) = path.split('.', 1)
        return CONNECTION.get_connection()[db][coll]