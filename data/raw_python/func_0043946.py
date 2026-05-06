def push(self, remote=None, revision=None):
        """
        Push changes from the local repository to a remote repository.

        :param remote: The location of a remote repository (a string or :data:`None`).
        :param revision: A specific revision to push (a string or :data:`None`).

        .. warning:: Depending on the version control backend the push command
                     may fail when there are no changes to push. No attempt has
                     been made to make this behavior consistent between
                     implementations (although the thought has crossed my
                     mind and I'll likely revisit this in the future).
        """
        # Make sure the local repository exists.
        self.ensure_exists()
        # Make sure there is a remote repository to push to.
        if not (remote or self.remote or self.default_push_remote):
            logger.info("Skipping push (no default remote is configured).")
        # Push the changes to the remote repository.
        timer = Timer()
        logger.info("Pushing changes from %s to %s ..",
                    format_path(self.local),
                    remote or self.remote or "default remote")
        self.context.execute(*self.get_push_command(remote, revision))
        logger.debug("Took %s to push changes to remote repository.", timer)