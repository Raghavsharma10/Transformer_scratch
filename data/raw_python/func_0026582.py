def broadcast(self, event):
        """Broadcasts an event either to all users or clients, depending on
        event flag"""
        try:
            if event.broadcasttype == "users":
                if len(self._users) > 0:
                    self.log("Broadcasting to all users:",
                             event.content, lvl=network)
                    for useruuid in self._users.keys():
                        self.fireEvent(
                            send(useruuid, event.content, sendtype="user"))
                        # else:
                        #    self.log("Not broadcasting, no users connected.",
                        #            lvl=debug)

            elif event.broadcasttype == "clients":
                if len(self._clients) > 0:
                    self.log("Broadcasting to all clients: ",
                             event.content, lvl=network)
                    for client in self._clients.values():
                        self.fireEvent(write(client.sock, event.content),
                                       "wsserver")
                        # else:
                        #    self.log("Not broadcasting, no clients
                        # connected.",
                        #            lvl=debug)
            elif event.broadcasttype == "socks":
                if len(self._sockets) > 0:
                    self.log("Emergency?! Broadcasting to all sockets: ",
                             event.content)
                    for sock in self._sockets:
                        self.fireEvent(write(sock, event.content), "wsserver")
                        # else:
                        #    self.log("Not broadcasting, no sockets
                        # connected.",
                        #            lvl=debug)

        except Exception as e:
            self.log("Error during broadcast: ", e, type(e), lvl=critical)