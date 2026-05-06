def _connect(self):
        """
        Connects to the server defined in the constructor.
        """

        self.first_data_sent_complete = False
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))

        msg = cr.Message()
        msg.type = cr.CONNECT
        msg.request_connect.auth_code = self.auth_code or 0
        msg.request_connect.send_playlist_songs = False
        msg.request_connect.downloader = False

        self.send_message(msg)