def register_db(cls, dbname):
        """Register method to keep list of dbs."""
        def decorator(subclass):
            """Register as decorator function."""
            cls._dbs[dbname] = subclass
            subclass.name = dbname
            return subclass
        return decorator