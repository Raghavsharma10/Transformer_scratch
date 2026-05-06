def run(self):
        """Run the sync.

        Confront the local and the remote directories and perform the needed changes."""

        # Check if remote path is present
        try:
            self.sftp.stat(self.remote_path)
        except FileNotFoundError as e:
            if self.create_remote_directory:
                self.sftp.mkdir(self.remote_path)
                self.logger.info(
                    "Created missing remote dir: '" + self.remote_path + "'")
            else:
                self.logger.error(
                    "Remote folder does not exists. "
                    "Add '-r' to create it if missing.")
                sys.exit(1)

        try:
            if self.delete:
                # First check for items to be removed
                self.check_for_deletion()

            # Now scan local for items to upload/create
            self.check_for_upload_create()
        except FileNotFoundError:
            # If this happens, probably the remote folder doesn't exist.
            self.logger.error(
                "Error while opening remote folder. Are you sure it does exist?")
            sys.exit(1)