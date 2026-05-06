def merge_dupes(self):
        """Merge two entries that correspond to the same entry."""
        for dupe in self.dupe_of:
            if dupe in self.catalog.entries:
                if self.catalog.entries[dupe]._stub:
                    # merge = False to avoid infinite recursion
                    self.catalog.load_entry_from_name(
                        dupe, delete=True, merge=False)
                self.catalog.copy_entry_to_entry(self.catalog.entries[dupe],
                                                 self)
                del self.catalog.entries[dupe]
        self.dupe_of = []