def run_async(self):
        """
        Spawns a new thread that runs the message loop until the Pebble disconnects.
        ``run_async`` will call :meth:`fetch_watch_info` on your behalf, and block until it receives a response.
        """
        thread = threading.Thread(target=self.run_sync)
        thread.daemon = True
        thread.name = "PebbleConnection"
        thread.start()
        self.fetch_watch_info()