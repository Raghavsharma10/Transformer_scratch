def status(self, remote=False):
        """
        Return the connection status, both locally and remotely.

        The local connection status is a dictionary that gives:
        * the count of multiple queries sent to the server.
        * the count of single queries sent to the server.
        * the count of actions sent to the server.
        * the count of actions executed successfully by the server.
        * the count of actions queued to go to the server.

        The remote connection status includes whether the server is live,
        as well as data about version and build.  The server data is
        cached, unless the remote flag is specified.

        :param remote: whether to query the server for its latest status
        :return: tuple of status dicts: (local, server).
        """
        if remote:
            components = urlparse.urlparse(self.endpoint)
            try:
                result = self.session.get(components[0] + "://" + components[1] + "/status", timeout=self.timeout)
            except Exception as e:
                if self.logger: self.logger.debug("Failed to connect to server for status: %s", e)
                result = None
            if result and result.status_code == 200:
                self.server_status = result.json()
                self.server_status["endpoint"] = self.endpoint
            elif result:
                if self.logger: self.logger.debug("Server status response not understandable: Status: %d, Body: %s",
                                                  result.status_code, result.text)
                self.server_status = {"endpoint": self.endpoint,
                                      "status": ("Unexpected HTTP status " + str(result.status_code) + " at: " +
                                                 strftime("%d %b %Y %H:%M:%S +0000", gmtime()))}
            else:
                self.server_status = {"endpoint": self.endpoint,
                                      "status": "Unreachable at: " + strftime("%d %b %Y %H:%M:%S +0000", gmtime())}
        return self.local_status, self.server_status