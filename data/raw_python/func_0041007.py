def create_fpath_dir(self, fpath: str):
        """Creates directory for fpath."""
        os.makedirs(os.path.dirname(fpath), exist_ok=True)