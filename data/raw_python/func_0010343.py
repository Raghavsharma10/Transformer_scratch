def start(self):
        """Creates a TCP connection to Device Cloud and sends a ConnectionRequest message"""
        self.log.info("Starting Insecure Session for Monitor %s" % self.monitor_id)
        if self.socket is not None:
            raise Exception("Socket already established for %s." % self)

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.client.hostname, PUSH_OPEN_PORT))
            self.socket.setblocking(0)
        except socket.error as exception:
            self.socket.close()
            self.socket = None
            raise

        self.send_connection_request()