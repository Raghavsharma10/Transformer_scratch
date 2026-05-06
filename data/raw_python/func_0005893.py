def set_basic_params(self, use_credentials=None, stats_server=None):
        """
        :param str|unicode use_credentials: Enable check of SCM_CREDENTIALS for tuntap client/server.

        :param str|unicode stats_server: Router stats server address to run at.

        """
        self._set_aliased('use-credentials', use_credentials)
        self._set_aliased('router-stats', stats_server)

        return self