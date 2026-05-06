def relative_root_dir(self):
        """Build the relative root dir path for the bundle version."""
        return Path(self.bundle.name) / str(self.created_at.date())