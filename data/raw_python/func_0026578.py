def disconnect(self, sock):
        """Handles socket disconnections"""

        self.log("Disconnect ", sock, lvl=debug)

        try:
            if sock in self._sockets:
                self.log("Getting socket", lvl=debug)
                sockobj = self._sockets[sock]
                self.log("Getting clientuuid", lvl=debug)
                clientuuid = sockobj.clientuuid
                self.log("getting useruuid", lvl=debug)
                useruuid = self._clients[clientuuid].useruuid

                self.log("Firing disconnect event", lvl=debug)
                self.fireEvent(clientdisconnect(clientuuid, self._clients[
                    clientuuid].useruuid))

                self.log("Logging out relevant client", lvl=debug)
                if useruuid is not None:
                    self.log("Client was logged in", lvl=debug)
                    try:
                        self._logoutclient(useruuid, clientuuid)
                        self.log("Client logged out", useruuid, clientuuid)
                    except Exception as e:
                        self.log("Couldn't clean up logged in user! ",
                                 self._users[useruuid], e, type(e),
                                 lvl=critical)
                self.log("Deleting Client (", self._clients.keys, ")",
                         lvl=debug)
                del self._clients[clientuuid]
                self.log("Deleting Socket", lvl=debug)
                del self._sockets[sock]
        except Exception as e:
            self.log("Error during disconnect handling: ", e, type(e),
                     lvl=critical)