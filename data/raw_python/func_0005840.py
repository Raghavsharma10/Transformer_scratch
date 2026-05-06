def set_on_exit_params(self, skip_hooks=None, skip_teardown=None):
        """Set params related to process exit procedure.

        :param bool skip_hooks: Skip ``EXIT`` phase hook.

            .. note:: Ignored by the master.

        :param bool skip_teardown: Allows skipping teardown (finalization) processes for some plugins.

            .. note:: Ignored by the master.

            Supported by:
                * Perl
                * Python

        """
        self._set('skip-atexit', skip_hooks, cast=bool)
        self._set('skip-atexit-teardown', skip_teardown, cast=bool)

        return self._section