def add(self, name, value):
        """
        Adds a new entry to the table

        We reduce the table size if the entry will make the
        table size greater than maxsize.
        """
        # We just clear the table if the entry is too big
        size = table_entry_size(name, value)
        if size > self._maxsize:
            self.dynamic_entries.clear()
            self._current_size = 0

        # Add new entry if the table actually has a size
        elif self._maxsize > 0:
            self.dynamic_entries.appendleft((name, value))
            self._current_size += size
            self._shrink()