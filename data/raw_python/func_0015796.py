def _ensure_file_path(self):
        """
        Ensure the storage path exists.
        If it doesn't, create it with "go-rwx" permissions.
        """
        storage_root = os.path.dirname(self.file_path)
        needs_storage_root = storage_root and not os.path.isdir(storage_root)
        if needs_storage_root:  # pragma: no cover
            os.makedirs(storage_root)
        if not os.path.isfile(self.file_path):
            # create the file without group/world permissions
            with open(self.file_path, 'w'):
                pass
            user_read_write = 0o600
            os.chmod(self.file_path, user_read_write)