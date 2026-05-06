def _on_connect_error(self, sock, err):  # pylint: disable=unused-argument
        """Error received from websocket"""
        if isinstance(err, SystemExit):
            self.log.error(f"Shutting down websocket connection")
        else:
            self.log.error(f"Websocket error: {err}")