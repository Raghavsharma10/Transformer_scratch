def _on_set_auth(self, sock, token):
        """Set Auth request received from websocket"""
        self.log.info(f"Token received: {token}")
        sock.setAuthtoken(token)