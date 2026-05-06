def connect(self, url=c.LOCALHOST, port=None, timeout=c.INITIAL_TIMEOUT,
                      debug=False):
        """socket connect to an already running starcraft2 process"""
        if port != None: # force a selection to a new port
            if self._port!=None: # if previously allocated port, return it
                portpicker.return_port(self._port)
            self._port = port
        elif self._port==None: # no connection exists
            self._port = portpicker.pick_unused_port()
        self._url = url
        if ":" in url and not url.startswith("["):  # Support ipv6 addresses.
            url = "[%s]" % url
        for i in range(timeout):
            startTime = time.time()
            if debug:
                print("attempt #%d to websocket connect to %s:%s"%(i, url, port))
            try:
                finalUrl = "ws://%s:%s/sc2api" %(url, self._port)
                ws = websocket.create_connection(finalUrl, timeout=timeout)
                #print("ws:", ws)
                self._client = protocol.StarcraftProtocol(ws)
                #super(ClientController, self).__init__(client) # ensure RemoteController initializtion is performed
                #if self.ping(): print("init ping()") # ensure the latest state is synced
                # ping returns:
                #   game_version:   "4.1.2.60604"
                #   data_version:   "33D9FE28909573253B7FC352CE7AEA40"
                #   data_build:     60604
                #   base_build:     60321
                return self
            except socket.error: pass  # SC2 hasn't started listening yet.
            except websocket.WebSocketException as err:
                print(err, type(err))
                if "Handshake Status 404" in str(err):
                    pass  # SC2 is listening, but hasn't set up the /sc2api endpoint yet.
                else: raise
            except Exception as e:
                print(type(e), e)
            sleepTime = max(0, 1 - (time.time() - startTime)) # try to wait for up to 1 second total
            if sleepTime:   time.sleep(sleepTime)
        raise websocket.WebSocketException("Could not connect to game at %s on port %s"%(url, port))