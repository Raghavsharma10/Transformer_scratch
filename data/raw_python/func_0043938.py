def export(self, directory, revision=None):
        """
        Export the complete tree from the local version control repository.

        :param directory: The directory where the tree should be exported
                          (a string).
        :param revision: The revision to export (a string or :data:`None`,
                         defaults to :attr:`default_revision`).
        """
        # Make sure we're dealing with an absolute pathname (because a relative
        # pathname would be interpreted as relative to the repository's main
        # directory, which isn't necessarily what the caller expects).
        directory = os.path.abspath(directory)
        # Make sure the local repository exists.
        self.create()
        # Export the tree from the local repository.
        timer = Timer()
        revision = revision or self.default_revision
        logger.info("Exporting revision '%s' in %s to %s ..", revision, format_path(self.local), directory)
        self.context.execute('mkdir', '-p', directory)
        self.context.execute(*self.get_export_command(directory, revision))
        logger.debug("Took %s to pull changes from remote %s repository.", timer, self.friendly_name)