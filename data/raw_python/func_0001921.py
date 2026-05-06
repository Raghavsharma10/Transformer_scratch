def start(self, blocking=False):
        """
        Start the interface

        :param blocking: Should the call block until stop() is called
            (default: False)
        :type blocking: bool
        :rtype: None
        """
        super(StartStopable, self).start()
        self._is_running = True
        # blocking
        try:
            while blocking and self._is_running:
                time.sleep(self._start_block_timeout)
        except IOError as e:
            if not str(e).lower().startswith("[errno 4]"):
                raise