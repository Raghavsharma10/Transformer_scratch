def rename(self, new_id):
        """
        Renames the DatabaseObject to have ID_KEY new_id.  This is the only
        way allowed by DatabaseObject to change the ID_KEY of an object.
        Trying to modify ID_KEY in the dictionary will raise an exception.
        
        @param new_id: the new value for ID_KEY
        
        NOTE: This is actually a create and delete.
        
        WARNING: If the system fails during a rename, data may be duplicated.
        """
        old_id = dict.__getitem__(self, ID_KEY)
        dict.__setitem__(self, ID_KEY, new_id)
        self._collection.save(self)
        self._collection.remove({ID_KEY: old_id})