def dict_merge(self, merge_to, merge_in):
        '''
        Recursively merges two dicts

        Overwrites any non-dictionary items
        merge_to <- merge_in
        Modifies merge_to dictionary

        @param merge_to: Base dictionary to merge into
        @param merge_in: Dictionary that may overwrite elements in merge_in
        '''
        for key, value in merge_in.items():
            # Just add, if the key doesn't exist yet
            # Or if set to None/Null
            if key not in merge_to.keys() or merge_to[key] is None:
                merge_to[key] = copy.copy(value)
                continue

            # Overwrite case, check for types
            # Make sure types are matching
            if not isinstance(value, type(merge_to[key])):
                raise MergeException('Types do not match! {}: {} != {}'.format(key, type(value), type(merge_to[key])))

            # Check if this is a dictionary item, in which case recursively merge
            if isinstance(value, dict):
                self.dict_merge(merge_to[key], value)
                continue

            # Otherwise just overwrite
            merge_to[key] = copy.copy(value)