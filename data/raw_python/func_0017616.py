def mount_point(self):
        """
        The pathname of the mount point of :attr:`directory` (a string or :data:`None`).

        If the ``stat --format=%m ...`` command that is used to determine the
        mount point fails, the value of this property defaults to :data:`None`.
        This enables graceful degradation on e.g. Mac OS X whose ``stat``
        implementation is rather bare bones compared to GNU/Linux.
        """
        try:
            return self.context.capture('stat', '--format=%m', self.directory, silent=True)
        except ExternalCommandFailed:
            return None