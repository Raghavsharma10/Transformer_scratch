def add_entry(self, name, load=True, delete=True):
        """Find an existing entry in, or add a new one to, the `entries` dict.

        FIX: rename to `create_entry`???

        Returns
        -------
        entries : OrderedDict of Entry objects
        newname : str
            Name of matching entry found in `entries`, or new entry added to
            `entries`
        """
        newname = self.clean_entry_name(name)

        if not newname:
            raise (ValueError('Fatal: Attempted to add entry with no name.'))

        # If entry already exists, return
        if newname in self.entries:
            self.log.debug("`newname`: '{}' (name: '{}') already exists.".
                           format(newname, name))
            # If this is a stub, we need to continue, possibly load file
            if self.entries[newname]._stub:
                self.log.debug("'{}' is a stub".format(newname))
            # If a full (non-stub) event exists, return its name
            else:
                self.log.debug("'{}' is not a stub, returning".format(newname))
                return newname

        # If entry is alias of another entry in `entries`, find and return that
        match_name = self.find_entry_name_of_alias(newname)
        if match_name is not None:
            self.log.debug(
                "`newname`: '{}' (name: '{}') already exists as alias for "
                "'{}'.".format(newname, name, match_name))
            newname = match_name

        # Load entry from file
        if load:
            loaded_name = self.load_entry_from_name(newname, delete=delete)
            if loaded_name:
                return loaded_name

        # If we match an existing event, return that
        if match_name is not None:
            return match_name

        # Create new entry
        new_entry = self.proto(catalog=self, name=newname)
        new_entry[self.proto._KEYS.SCHEMA] = self.SCHEMA.URL
        self.log.log(self.log._LOADED,
                     "Created new entry for '{}'".format(newname))
        # Add entry to dictionary
        self.entries[newname] = new_entry
        return newname