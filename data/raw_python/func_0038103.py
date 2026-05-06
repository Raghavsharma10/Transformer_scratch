def start(self):
        """
        start
        """
        def main_thread():
            # create resp, req thread pool
            self._create_thread_pool()
            # start connection, this will block until stop()
            self.conn_thread = Thread(target=self._conn.connect)
            self.conn_thread.daemon = True
            self.conn_thread.start()

            # register model to controller...
            self.is_ready.wait()

            if hasattr(self, 'run'):
                _logger.debug("Start running...")
                self.run()

        # start main_thread
        self.main_thread = Thread(target=main_thread)
        self.main_thread.daemon = True
        self.main_thread.start()

        if threading.current_thread().__class__.__name__ == '_MainThread':
            # control this bundle stop or not
            while not self.stop_event.wait(1):
                sleep(1)
        else:
            self.stop_event.wait()

        self.stop()
        _logger.debug("Shutdown successfully")