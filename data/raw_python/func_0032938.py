def _recv_callback(self, msg):
        """
        Method is called when there is a message coming from a Mongrel2 server.
        This message should be a valid Request String.
        """
        m2req = MongrelRequest.parse(msg[0])
        MongrelConnection(m2req, self._sending_stream, self.request_callback,
            no_keep_alive=self.no_keep_alive, xheaders=self.xheaders)