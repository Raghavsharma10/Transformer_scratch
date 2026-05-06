def save(self):
        """
        Saves the current state of the DatabaseObject to the database.  Fills
        in missing values from defaults before saving.
        
        NOTE: The actual operation here is to overwrite the entry in the
        database with the same ID_KEY.
        
        WARNING: While the save operation itself is atomic, it is not atomic
        with loads and modifications to the object.  You must provide your own
        synchronization if you have multiple threads or processes possibly
        modifying the same database object.  The update method is better from
        a concurrency perspective.
        
        @raise MalformedObjectError: if the object does not provide a value
            for a REQUIRED default
        """
        self._pre_save()
        self._collection.replace_one({ID_KEY: self[ID_KEY]}, dict(self))