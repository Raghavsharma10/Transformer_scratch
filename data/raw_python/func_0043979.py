def iter_files(self):
        """Iterate over files."""
        # file_iter may be a callable or an iterator
        if callable(self.file_iter):
            return self.file_iter()
        return iter(self.file_iter)