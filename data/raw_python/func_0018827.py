def GET_close_server(self) -> None:
        """Stop and close the *HydPy* server."""
        def _close_server():
            self.server.shutdown()
            self.server.server_close()
        shutter = threading.Thread(target=_close_server)
        shutter.deamon = True
        shutter.start()