def _logoutclient(self, useruuid, clientuuid):
        """Log out a client and possibly associated user"""

        self.log("Cleaning up client of logged in user.", lvl=debug)
        try:
            self._users[useruuid].clients.remove(clientuuid)
            if len(self._users[useruuid].clients) == 0:
                self.log("Last client of user disconnected.", lvl=verbose)

                self.fireEvent(userlogout(useruuid, clientuuid))
                del self._users[useruuid]

            self._clients[clientuuid].useruuid = None
        except Exception as e:
            self.log("Error during client logout: ", e, type(e),
                     clientuuid, useruuid, lvl=error,
                     exc=True)