def set_reload_params(self, mercy=None, exit=None):
        """Set reload related params.

        :param int mercy: Set the maximum time (in seconds) we wait
            for workers and other processes to die during reload/shutdown.

        :param bool exit: Force exit even if a reload is requested.

        """
        self._set('reload-mercy', mercy)
        self.set_exit_events(reload=exit)

        return self._section