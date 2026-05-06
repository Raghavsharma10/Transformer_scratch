def update(self, update_dict=None, raw=False, **kwargs):
        """
        Applies updates both to the database object and to the database via the
        mongo update method with the $set argument.  Use the `raw` keyword to
        perform an arbitrary mongo update query.
        
        WARNING: Raw updates do not perform type checking.
        
        WARNING: While the update operation itself is atomic, it is not atomic
        with loads and modifications to the object.  You must provide your own
        synchronization if you have multiple threads or processes possibly
        modifying the same database object.  While this is safer from a
        concurrency perspective than the access pattern load -> modify -> save
        as it only updates keys specified in the update_dict, it will still
        overwrite updates to those same keys that were made while the object
        was held in memory.
        
        @param update_dict: dictionary of updates to apply
        @param raw: if set to True, uses the contents of update_dict directly
            to perform the update rather than wrapping them in $set.
        @param **kwargs: used as update_dict if no update_dict is None
        """
        if update_dict is None:
            update_dict = kwargs
        if raw:
            self._collection.update_one({ID_KEY: self[ID_KEY]}, update_dict)
            new_data = self._collection.find_one({ID_KEY: self[ID_KEY]})
            dict.clear(self)
            dict.update(self, new_data)
        else:
            for key, value in update_dict.items():
                self._check_type(key, value)
            dict.update(self, update_dict)
            self._collection.update_one({ID_KEY: self[ID_KEY]}, {SET: update_dict})