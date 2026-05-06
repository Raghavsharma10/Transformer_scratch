def blob_handler(self, cmd):
        """Process a BlobCommand."""
        # These never pass through directly. We buffer them and only
        # output them if referenced by an interesting command.
        self.blobs[cmd.id] = cmd
        self.keep = False