def find_revision_number(self, revision=None):
        """Find the local revision number of the given revision."""
        # Make sure the local repository exists.
        self.create()
        # Try to find the revision number of the specified revision.
        revision = self.expand_branch_name(revision)
        output = self.context.capture('git', 'rev-list', revision, '--count')
        if not (output and output.isdigit()):
            msg = "Failed to find local revision number! ('git rev-list --count' gave unexpected output)"
            raise ValueError(msg)
        return int(output)