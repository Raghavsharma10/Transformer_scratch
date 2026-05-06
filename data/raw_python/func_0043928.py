def create(self):
        """
        Create the local repository (if it doesn't already exist).

        :returns: :data:`True` if the local repository was just created,
                  :data:`False` if it already existed.

        What :func:`create()` does depends on the situation:

        - When :attr:`exists` is :data:`True` nothing is done.
        - When the :attr:`local` repository doesn't exist but a :attr:`remote`
          repository location is given, a clone of the remote repository is
          created.
        - When the :attr:`local` repository doesn't exist and no :attr:`remote`
          repository has been specified then a new local repository will be
          created.

        When :func:`create()` is responsible for creating the :attr:`local`
        repository it will make sure the :attr:`bare` option is respected.
        """
        if self.exists:
            logger.debug("Local %s repository (%s) already exists, ignoring request to create it.",
                         self.friendly_name, format_path(self.local))
            return False
        else:
            timer = Timer()
            if self.remote:
                logger.info("Creating local %s repository (%s) by cloning %s ..",
                            self.friendly_name, format_path(self.local), self.remote)
            else:
                logger.info("Creating local %s repository (%s) ..",
                            self.friendly_name, format_path(self.local))
            self.context.execute(*self.get_create_command())
            logger.debug("Took %s to %s local %s repository.",
                         timer, "clone" if self.remote else "create",
                         self.friendly_name)
            if self.remote:
                self.mark_updated()
            # Ensure that all further commands are executed in the local repository.
            self.update_context()
            return True