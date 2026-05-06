def send(self, event):
        """Sends a packet to an already known user or one of his clients by
        UUID"""

        try:
            jsonpacket = json.dumps(event.packet, cls=ComplexEncoder)
            if event.sendtype == "user":
                # TODO: I think, caching a user name <-> uuid table would
                # make sense instead of looking this up all the time.

                if event.uuid is None:
                    userobject = objectmodels['user'].find_one({
                        'name': event.username
                    })
                else:
                    userobject = objectmodels['user'].find_one({
                        'uuid': event.uuid
                    })

                if userobject is None:
                    self.log("No user by that name known.", lvl=warn)
                    return
                else:
                    uuid = userobject.uuid

                self.log("Broadcasting to all of users clients: '%s': '%s" % (
                    uuid, str(event.packet)[:20]), lvl=network)
                if uuid not in self._users:
                    self.log("User not connected!", event, lvl=critical)
                    return
                clients = self._users[uuid].clients

                for clientuuid in clients:
                    sock = self._clients[clientuuid].sock

                    if not event.raw:
                        self.log("Sending json to client", jsonpacket[:50],
                                 lvl=network)

                        self.fireEvent(write(sock, jsonpacket), "wsserver")
                    else:
                        self.log("Sending raw data to client")
                        self.fireEvent(write(sock, event.packet), "wsserver")
            else:  # only to client
                self.log("Sending to user's client: '%s': '%s'" % (
                    event.uuid, jsonpacket[:20]), lvl=network)
                if event.uuid not in self._clients:
                    if not event.fail_quiet:
                        self.log("Unknown client!", event.uuid, lvl=critical)
                        self.log("Clients:", self._clients, lvl=debug)
                    return

                sock = self._clients[event.uuid].sock
                if not event.raw:
                    self.fireEvent(write(sock, jsonpacket), "wsserver")
                else:
                    self.log("Sending raw data to client", lvl=network)
                    self.fireEvent(write(sock, event.packet[:20]), "wsserver")

        except Exception as e:
            self.log("Exception during sending: %s (%s)" % (e, type(e)),
                     lvl=critical, exc=True)