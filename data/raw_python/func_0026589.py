def read(self, *args):
        """Handles raw client requests and distributes them to the
        appropriate components"""

        self.log("Beginning new transaction: ", args, lvl=network)
        try:
            sock, msg = args[0], args[1]
            user = password = client = clientuuid = useruuid = requestdata = \
                requestaction = None
            # self.log("", msg)

            clientuuid = self._sockets[sock].clientuuid
        except Exception as e:
            self.log("Receiving error: ", e, type(e), lvl=error)

        if clientuuid in self._flooding:
            return

        try:
            msg = json.loads(msg)
            self.log("Message from client received: ", msg, lvl=network)
        except Exception as e:
            self.log("JSON Decoding failed! %s (%s of %s)" % (msg, e, type(e)))
            return

        try:
            requestcomponent = msg['component']
            requestaction = msg['action']
        except (KeyError, AttributeError) as e:
            self.log("Unpacking error: ", msg, e, type(e), lvl=error)
            return

        if self._check_flood_protection(requestcomponent, requestaction,
                                        clientuuid):
            self.log('Flood protection triggered')
            self._flooding[clientuuid] = time()

        try:
            # TODO: Do not unpickle or decode anything from unsafe events
            requestdata = msg['data']
            if isinstance(requestdata, (dict, list)) and 'raw' in requestdata:
                # self.log(requestdata['raw'], lvl=critical)
                requestdata['raw'] = b64decode(requestdata['raw'])
                # self.log(requestdata['raw'])
        except (KeyError, AttributeError) as e:
            self.log("No payload.", lvl=network)
            requestdata = None

        if requestcomponent == "auth":
            self._handleAuthenticationEvents(requestdata, requestaction,
                                             clientuuid, sock)
            return

        try:
            client = self._clients[clientuuid]
        except KeyError as e:
            self.log('Could not get client for request!', e, type(e), lvl=warn)
            return

        if requestcomponent in self.anonymous_events and requestaction in \
            self.anonymous_events[requestcomponent]:
            self.log('Executing anonymous event:', requestcomponent,
                     requestaction)
            try:
                self._handleAnonymousEvents(requestcomponent, requestaction,
                                            requestdata, client)
            except Exception as e:
                self.log("Anonymous request failed:", e, type(e), lvl=warn,
                         exc=True)
            return

        elif requestcomponent in self.authorized_events:
            try:
                useruuid = client.useruuid
                self.log("Authenticated operation requested by ",
                         useruuid, client.config, lvl=network)
            except Exception as e:
                self.log("No useruuid!", e, type(e), lvl=critical)
                return

            self.log('Checking if user is logged in', lvl=verbose)

            try:
                user = self._users[useruuid]
            except KeyError:
                if not (requestaction == 'ping' and requestcomponent == 'hfos.ui.clientmanager'):
                    self.log("User not logged in.", lvl=warn)

                return

            self.log('Handling event:', requestcomponent, requestaction, lvl=verbose)
            try:
                self._handleAuthorizedEvents(requestcomponent, requestaction,
                                             requestdata, user, client)
            except Exception as e:
                self.log("User request failed: ", e, type(e), lvl=warn,
                         exc=True)
        else:
            self.log('Invalid event received:', requestcomponent, requestaction, lvl=warn)