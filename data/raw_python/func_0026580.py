def connect(self, *args):
        """Registers new sockets and their clients and allocates uuids"""

        self.log("Connect ", args, lvl=verbose)

        try:
            sock = args[0]
            ip = args[1]

            if sock not in self._sockets:
                self.log("New client connected:", ip, lvl=debug)
                clientuuid = str(uuid4())
                self._sockets[sock] = Socket(ip, clientuuid)
                # Key uuid is temporary, until signin, will then be replaced
                #  with account uuid

                self._clients[clientuuid] = Client(
                    sock=sock,
                    ip=ip,
                    clientuuid=clientuuid,
                )

                self.log("Client connected:", clientuuid, lvl=debug)
            else:
                self.log("Old IP reconnected!", lvl=warn)
                #     self.fireEvent(write(sock, "Another client is
                # connecting from your IP!"))
                #     self._sockets[sock] = (ip, uuid.uuid4())
        except Exception as e:
            self.log("Error during connect: ", e, type(e), lvl=critical)