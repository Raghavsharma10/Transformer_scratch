def _move(self, new_path):
        """
        Moves the collection to a different database path
        
        NOTE: this function is intended for command prompt use only.
        
        WARNING: if execution is interrupted halfway through, the collection will
        be split into multiple pieces.  Furthermore, there is a possible
        duplication of the database object being processed at the time of
        interruption.
        
        @param new_path: the new place for the collection to live, in the format
            "database.collection"
        """
        for elt in self:
            DatabaseObject.create(elt, path=new_path)
        for elt in self:
            elt._collection.remove({ID_KEY: elt[ID_KEY]})