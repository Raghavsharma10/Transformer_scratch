def remove(self):
        """
        Deletes the object from the database
        
        WARNING: This cannot be undone.  Be really careful when deleting
        programatically.  It is recommended to backup your database before
        applying specific deletes.  If your application uses deletes regularly,
        it is strongly recommended that you have a recurring backup system.
        """
        self._collection.remove({ID_KEY: self[ID_KEY]})
        dict.clear(self)