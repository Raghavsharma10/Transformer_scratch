def context(self):
        """
        An execution context created using :mod:`executor.contexts`.

        The value of :attr:`context` defaults to a
        :class:`~executor.contexts.LocalContext` object with the following
        characteristics:

        - The working directory of the execution context is set to the
          value of :attr:`directory`.

        - The environment variable given by :data:`DIRECTORY_VARIABLE` is set
          to the value of :attr:`directory`.

        :raises: :exc:`.MissingPasswordStoreError` when :attr:`directory`
                 doesn't exist.
        """
        # Make sure the directory exists.
        self.ensure_directory_exists()
        # Prepare the environment variables.
        environment = {DIRECTORY_VARIABLE: self.directory}
        try:
            # Try to enable the GPG agent in headless sessions.
            environment.update(get_gpg_variables())
        except Exception:
            # If we failed then let's at least make sure that the
            # $GPG_TTY environment variable is set correctly.
            environment.update(GPG_TTY=execute("tty", capture=True, check=False, tty=True, silent=True))
        return LocalContext(directory=self.directory, environment=environment)