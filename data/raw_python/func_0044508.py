def blob_handler(self, cmd):
        """Process a BlobCommand."""
        self.cmd_counts[cmd.name] += 1
        if cmd.mark is None:
            self.blobs['unmarked'].add(cmd.id)
        else:
            self.blobs['new'].add(cmd.id)
            # Marks can be re-used so remove it from used if already there.
            # Note: we definitely do NOT want to remove it from multi if
            # it's already in that set.
            try:
                self.blobs['used'].remove(cmd.id)
            except KeyError:
                pass