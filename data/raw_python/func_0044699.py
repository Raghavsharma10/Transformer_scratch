def full_path(self):
        """Return the full path to the file."""
        if Path(self.path).is_absolute():
            return self.path
        else:
            return str(self.app_root / self.path)