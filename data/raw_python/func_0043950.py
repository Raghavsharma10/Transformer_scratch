def update_context(self):
        """
        Try to ensure that external commands are executed in the local repository.

        What :func:`update_context()` does depends on whether the directory
        given by :attr:`local` exists:

        - If :attr:`local` exists then the working directory of :attr:`context`
          will be set to :attr:`local`. This is to ensure that version control
          commands are run inside of the intended version control repository.

        - If :attr:`local` doesn't exist then the working directory of
          :attr:`context` is cleared. This avoids external commands from
          failing due to an invalid (non existing) working directory.
        """
        if self.context.is_directory(self.local):
            # Set the working directory of the execution context
            # to the directory containing the local repository.
            self.context.options['directory'] = self.local
        else:
            # Clear the execution context's working directory.
            self.context.options.pop('directory', None)