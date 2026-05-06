def get_by_index(self, index):
        """
        Returns the entry specified by index

        Note that the table is 1-based ie an index of 0 is
        invalid.  This is due to the fact that a zero value
        index signals that a completely unindexed header
        follows.

        The entry will either be from the static table or
        the dynamic table depending on the value of index.
        """
        index -= 1
        if 0 <= index < len(CocaineHeaders.STATIC_TABLE):
            return CocaineHeaders.STATIC_TABLE[index]
        index -= len(CocaineHeaders.STATIC_TABLE)
        if 0 <= index < len(self.dynamic_entries):
            return self.dynamic_entries[index]
        raise InvalidTableIndex("Invalid table index %d" % index)