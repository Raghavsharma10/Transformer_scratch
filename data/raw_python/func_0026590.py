def authentication(self, event):
        """Links the client to the granted account and profile,
        then notifies the client"""

        try:
            self.log("Authorization has been granted by DB check:",
                     event.username, lvl=debug)

            account, profile, clientconfig = event.userdata

            useruuid = event.useruuid
            originatingclientuuid = event.clientuuid
            clientuuid = clientconfig.uuid

            if clientuuid != originatingclientuuid:
                self.log("Mutating client uuid to request id:",
                         clientuuid, lvl=network)
            # Assign client to user
            if useruuid in self._users:
                signedinuser = self._users[useruuid]
            else:
                signedinuser = User(account, profile, useruuid)
                self._users[account.uuid] = signedinuser

            if clientuuid in signedinuser.clients:
                self.log("Client configuration already logged in.",
                         lvl=critical)
                # TODO: What now??
                # Probably senseful would be to add the socket to the
                # client's other socket
                # The clients would be identical then - that could cause
                # problems
                # which could be remedied by duplicating the configuration
            else:
                signedinuser.clients.append(clientuuid)
                self.log("Active client (", clientuuid, ") registered to "
                                                        "user", useruuid,
                         lvl=debug)

            # Update socket..
            socket = self._sockets[event.sock]
            socket.clientuuid = clientuuid
            self._sockets[event.sock] = socket

            # ..and client lists

            try:
                language = clientconfig.language
            except AttributeError:
                language = "en"

            # TODO: Rewrite and simplify this:
            newclient = Client(
                sock=event.sock,
                ip=socket.ip,
                clientuuid=clientuuid,
                useruuid=useruuid,
                name=clientconfig.name,
                config=clientconfig,
                language=language
            )

            del (self._clients[originatingclientuuid])
            self._clients[clientuuid] = newclient

            authpacket = {"component": "auth", "action": "login",
                          "data": account.serializablefields()}
            self.log("Transmitting Authorization to client", authpacket,
                     lvl=network)
            self.fireEvent(
                write(event.sock, json.dumps(authpacket)),
                "wsserver"
            )

            profilepacket = {"component": "profile", "action": "get",
                             "data": profile.serializablefields()}
            self.log("Transmitting Profile to client", profilepacket,
                     lvl=network)
            self.fireEvent(write(event.sock, json.dumps(profilepacket)),
                           "wsserver")

            clientconfigpacket = {"component": "clientconfig", "action": "get",
                                  "data": clientconfig.serializablefields()}
            self.log("Transmitting client configuration to client",
                     clientconfigpacket, lvl=network)
            self.fireEvent(write(event.sock, json.dumps(clientconfigpacket)),
                           "wsserver")

            self.fireEvent(userlogin(clientuuid, useruuid, clientconfig, signedinuser))

            self.log("User configured: Name",
                     signedinuser.account.name, "Profile",
                     signedinuser.profile.uuid, "Clients",
                     signedinuser.clients,
                     lvl=debug)

        except Exception as e:
            self.log("Error (%s, %s) during auth grant: %s" % (
                type(e), e, event), lvl=error)