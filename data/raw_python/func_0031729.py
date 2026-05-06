def shutdown_login_server(self, ):
        """Shutdown the login server and thread

        :returns: None
        :rtype: None
        :raises: None
        """
        log.debug('Shutting down the login server thread.')
        self.login_server.shutdown()
        self.login_server.server_close()
        self.login_thread.join()