def _add_cat_dict(self,
                      cat_dict_class,
                      key_in_self,
                      check_for_dupes=True,
                      **kwargs):
        """Add a CatDict to this Entry if initialization succeeds and it
        doesn't already exist within the Entry.
        """
        # Try to create a new instance of this subclass of `CatDict`
        new_entry = self._init_cat_dict(cat_dict_class, key_in_self, **kwargs)
        if new_entry is None:
            return False

        # Compare this new entry with all previous entries to make sure is new
        if cat_dict_class != Error:
            for item in self.get(key_in_self, []):
                if new_entry.is_duplicate_of(item):
                    item.append_sources_from(new_entry)
                    # Return the entry in case we want to use any additional
                    # tags to augment the old entry
                    return new_entry

        self.setdefault(key_in_self, []).append(new_entry)

        return True