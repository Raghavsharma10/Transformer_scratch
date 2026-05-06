def set_exit_events(self, no_workers=None, idle=None, reload=None, sig_term=None):
        """Do exit on certain events

        :param bool no_workers: Shutdown uWSGI when no workers are running.

        :param bool idle: Shutdown uWSGI when idle.

        :param bool reload: Force exit even if a reload is requested.

        :param bool sig_term: Exit on SIGTERM instead of brutal workers reload.

            .. note:: Before 2.1 SIGTERM reloaded the stack while SIGINT/SIGQUIT shut it down.

        """
        self._set('die-on-no-workers', no_workers, cast=bool)
        self._set('exit-on-reload', reload, cast=bool)
        self._set('die-on-term', sig_term, cast=bool)
        self.set_idle_params(exit=idle)

        return self._section