def _init_threads(self):
        """Initializes the IO and Writer threads"""
        if self._io_thread is None:
            self._io_thread = Thread(target=self._select)
            self._io_thread.start()

        if self._writer_thread is None:
            self._writer_thread = Thread(target=self._writer)
            self._writer_thread.start()