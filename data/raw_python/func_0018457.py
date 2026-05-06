def check_for_upload_create(self, relative_path=None):
        """Traverse the relative_path tree and check for files that need to be uploaded/created.

        Relativity here refers to the shared directory tree."""
        for f in os.listdir(
            path_join(
                self.local_path, relative_path) if relative_path else self.local_path
        ):
            self.node_check_for_upload_create(relative_path, f)