def start(self, *args, **kwargs):
        """
        Start to read the stream(s).
        """
        queue = Queue()
        stdout_reader, stderr_reader = \
            self._create_readers(queue, *args, **kwargs)

        self.thread = threading.Thread(target=self._read,
                                       args=(stdout_reader,
                                             stderr_reader,
                                             queue))
        self.thread.daemon = True
        self.thread.start()