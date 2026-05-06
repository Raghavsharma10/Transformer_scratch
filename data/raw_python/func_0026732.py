def _add_cat_dict(self,
                      cat_dict_class,
                      key_in_self,
                      check_for_dupes=True,
                      compare_to_existing=True,
                      **kwargs):
        """Add a `CatDict` to this `Entry`.

        CatDict only added if initialization succeeds and it
        doesn't already exist within the Entry.
        """
        # Make sure that a source is given, and is valid (nor erroneous)
        if cat_dict_class != Error:
            try:
                source = self._check_cat_dict_source(cat_dict_class,
                                                     key_in_self, **kwargs)
            except CatDictError as err:
                if err.warn:
                    self._log.info("'{}' Not adding '{}': '{}'".format(self[
                        self._KEYS.NAME], key_in_self, str(err)))
                return False

            if source is None:
                return False

        # Try to create a new instance of this subclass of `CatDict`
        new_entry = self._init_cat_dict(cat_dict_class, key_in_self, **kwargs)
        if new_entry is None:
            return False

        # Compare this new entry with all previous entries to make sure is new
        if compare_to_existing and cat_dict_class != Error:
            for item in self.get(key_in_self, []):
                if new_entry.is_duplicate_of(item):
                    item.append_sources_from(new_entry)
                    # Return the entry in case we want to use any additional
                    # tags to augment the old entry
                    return new_entry

        # If this is an alias, add it to the parent catalog's reverse
        # dictionary linking aliases to names for fast lookup.
        if key_in_self == self._KEYS.ALIAS:
            # Check if this adding this alias makes us a dupe, if so mark
            # ourselves as a dupe.
            if (check_for_dupes and 'aliases' in dir(self.catalog) and
                    new_entry[QUANTITY.VALUE] in self.catalog.aliases):
                possible_dupe = self.catalog.aliases[new_entry[QUANTITY.VALUE]]
                # print(possible_dupe)
                if (possible_dupe != self[self._KEYS.NAME] and
                        possible_dupe in self.catalog.entries):
                    self.dupe_of.append(possible_dupe)
            if 'aliases' in dir(self.catalog):
                self.catalog.aliases[new_entry[QUANTITY.VALUE]] = self[
                    self._KEYS.NAME]

        self.setdefault(key_in_self, []).append(new_entry)

        if (key_in_self == self._KEYS.ALIAS and check_for_dupes and
                self.dupe_of):
            self.merge_dupes()

        return True