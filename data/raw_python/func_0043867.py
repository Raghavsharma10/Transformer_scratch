def find_revision_id(self, revision=None):
        """Find the global revision id of the given revision."""
        # Make sure the local repository exists.
        self.create()
        # Try to find the revision id of the specified revision.
        revision = revision or self.default_revision
        output = self.context.capture(
            'bzr', 'version-info', '--revision=%s' % revision,
            '--custom', '--template={revision_id}',
        )
        # Validate the `bzr version-info' output.
        if not output:
            msg = "Failed to find global revision id! ('bzr version-info' gave unexpected output)"
            raise ValueError(msg)
        return output