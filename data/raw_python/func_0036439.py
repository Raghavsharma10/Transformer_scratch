def from_db(cls, db, force=False):
        """Make instance from database.

        For performance, this caches the episode types for the database.  The
        `force` parameter can be used to bypass this.

        """
        if force or db not in cls._cache:
            cls._cache[db] = cls._new_from_db(db)
        return cls._cache[db]