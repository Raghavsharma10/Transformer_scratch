def update_context(self):
        """
        Make sure Bazaar respects the configured author.

        This method first calls :func:`.Repository.update_context()` and then
        it sets the ``$BZR_EMAIL`` environment variable based on the value of
        :attr:`~Repository.author` (but only if :attr:`~Repository.author` was
        set by the caller).

        This is a workaround for a weird behavior of Bazaar that I've observed
        when running under Python 2.6: The ``bzr commit --author`` command line
        option is documented but it doesn't prevent Bazaar from nevertheless
        reporting the following error::

         bzr: ERROR: Unable to determine your name.
         Please, set your name with the 'whoami' command.
         E.g. bzr whoami "Your Name <name@example.com>"
        """
        # Call our superclass.
        super(BzrRepo, self).update_context()
        # Try to ensure that $BZR_EMAIL is set (see above for the reason)
        # but only if the `author' property was set by the caller (more
        # specifically there's no point in setting $BZR_EMAIL to the
        # output of `bzr whoami').
        if self.__dict__.get('author'):
            environment = self.context.options.setdefault('environment', {})
            environment.setdefault('BZR_EMAIL', self.author.combined)