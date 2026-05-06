def connect_ws(self, post_connect_callback, channels, reconnect=False):
        """
        Connect to a websocket
        :channels:  List of SockChannel instances
        """
        self.post_conn_cb = post_connect_callback
        self.channels = channels
        self.wsendpoint = self.context["conf"]["endpoints"].get("websocket")

        # Skip connecting if we don't have any channels to listen to
        if not channels:
            return

        # Create socket, connect, setting callbacks along the way
        self.sock = Socketcluster.socket(self.wsendpoint)
        self.sock.setBasicListener(self._on_connect, self._on_connect_close,
                                   self._on_connect_error)
        self.sock.setAuthenticationListener(self._on_set_auth, self._on_auth)
        self.sock.setreconnection(reconnect)
        self.sock.connect()