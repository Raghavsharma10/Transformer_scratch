def start(self):
        """
        Start animation thread.
        """
        self.thread = threading.Thread(target=self._animate)
        self.thread.start()
        return