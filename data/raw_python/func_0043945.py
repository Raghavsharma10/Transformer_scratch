def pull(self, remote=None, revision=None):
        """
        Pull changes from a remote repository into the local repository.

        :param remote: The location of a remote repository (a string or :data:`None`).
        :param revision: A specific revision to pull (a string or :data:`None`).

        If used in combination with :class:`limit_vcs_updates` this won't
        perform redundant updates.
        """
        remote = remote or self.remote
        # Make sure the local repository exists.
        if self.create() and (remote == self.remote or not remote):
            # Don't waste time pulling from a remote repository that we just cloned.
            logger.info("Skipping pull from default remote because we just created the local %s repository.",
                        self.friendly_name)
            return
        # Make sure there is a remote repository to pull from.
        if not (remote or self.default_pull_remote):
            logger.info("Skipping pull (no default remote is configured).")
            return
        # Check if we're about to perform a redundant pull.
        update_limit = int(os.environ.get(UPDATE_VARIABLE, '0'))
        if update_limit and self.last_updated >= update_limit:
            logger.info("Skipping pull due to update limit.")
            return
        # Pull the changes from the remote repository.
        timer = Timer()
        logger.info("Pulling changes from %s into local %s repository (%s) ..",
                    remote or "default remote", self.friendly_name, format_path(self.local))
        self.context.execute(*self.get_pull_command(remote=remote, revision=revision))
        logger.debug("Took %s to pull changes from remote %s repository.", timer, self.friendly_name)
        self.mark_updated()