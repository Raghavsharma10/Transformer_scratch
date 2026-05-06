def filtered_entries(self):
        """A list of :class:`PasswordEntry` objects that don't match the exclude list."""
        return [
            e for e in self.entries if not any(fnmatch.fnmatch(e.name.lower(), p.lower()) for p in self.exclude_list)
        ]