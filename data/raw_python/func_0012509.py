def start(self):
        """Create a background thread for httpd and serve 'forever'"""
        self._process = threading.Thread(target=self._background_runner)
        self._process.start()