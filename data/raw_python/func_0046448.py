def userinfo(self, username, realname=None):
        """Set the username and realname for this connection.

        Note: this should only be called once, on connect. (The default
        on-connect routine calls this automatically.)
        """
        realname = realname or username

        _log.info("Requesting user info update: username=%s realname=%s",
            username, realname)

        self.send("USER", username, socket.getfqdn(), self.server.host,
            ":%s" % realname) # Realname should always be prefixed by a colon
        self.user.username = username
        self.user.realname = realname