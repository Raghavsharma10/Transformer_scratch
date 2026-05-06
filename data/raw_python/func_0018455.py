def create_update_symlink(self, link_destination, remote_path):
        """Create a new link pointing to link_destination in remote_path position."""
        try:  # if there's anything, delete it
            self.sftp.remove(remote_path)
        except IOError:  # that's fine, nothing exists there!
            pass
        finally:  # and recreate the link
            try:
                self.sftp.symlink(link_destination, remote_path)
            except OSError as e:
                # Sometimes, if links are "too" different, symlink fails.
                # Sadly, nothing we can do about it.
                self.logger.error("error while symlinking {} to {}: {}".format(
                    remote_path, link_destination, e))