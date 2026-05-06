def create(cls, data, path=None, defaults=None, overwrite=False,
               random_id=False, **kwargs):
        """
        Creates a new database object and stores it in the database
        
        NOTE: The path and defaults parameters to this function are to allow
        use of the DatabaseObject class directly.  However, this class is
        intended for subclassing and children of it should override the PATH
        and DEFAULTS attributes rather than passing them as parameters here.
        
        @param data: dictionary of data that the object should be created with;
            this must follow all mongo rules, as well as have an entry for
            ID_KEY unless random_id == True
        @param path: the path of the database to use, in the form
            "database.collection"
        @param defaults: the defaults dictionary to use for this object
        @param overwrite: if set to true, will overwrite any object in the
            database with the same ID_KEY; if set to false will raise an
            exception if there is another object with the same ID_KEY
        @param random_id: stores the new object with a random value for ID_KEY;
            overwrites data[ID_KEY]
        @param **kwargs: ignored
        
        @raise Exception: if path and self.PATH are None; the database path
            must be defined in at least one of these
        @raise DatabaseConflictError: if there is already an object with that
            ID_KEY and overwrite == False
        @raise MalformedObjectError: if a REQUIRED key of defaults is missing,
            or if the ID_KEY of the object is None and random_id is False
        """
        self = cls(path=path, defaults=defaults, _new_object=data)
        for key, value in self.items():
            if key == ID_KEY:
                continue
            if self.DEFAULTS and key not in self.DEFAULTS:
                self._handle_non_default_key(key, value)
            self._check_type(key, value)
        if random_id and ID_KEY in self:
            dict.__delitem__(self, ID_KEY)
        if not random_id and ID_KEY not in self:
            raise MalformedObjectError("No " + ID_KEY + " key in item")
        if not random_id and not overwrite and self._collection.find_one({ID_KEY: data[ID_KEY]}):
            raise DatabaseConflictError('ID_KEY "%s" already exists in collection %s' %
                                        (data[ID_KEY], self.PATH))
        self._pre_save()
        if ID_KEY in self and overwrite:
            self._collection.replace_one({ID_KEY: self[ID_KEY]}, dict(self), upsert=True)
        else:
            insert_result = self._collection.insert_one(dict(self))
            dict.__setitem__(self, ID_KEY, insert_result.inserted_id)
        return self