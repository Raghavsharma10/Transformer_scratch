def start(self, interval=None, iterations=None):
        """Start the timer.

        A timeout event will be generated every *interval* seconds.
        If *interval* is None, then self.interval will be used.

        If *iterations* is specified, the timer will stop after
        emitting that number of events. If unspecified, then
        the previous value of self.iterations will be used. If the value is
        negative, then the timer will continue running until stop() is called.

        If the timer is already running when this function is called, nothing
        happens (timer continues running as it did previously, without
        changing the interval, number of iterations, or emitting a timer
        start event).
        """
        if self.running:
            return  # don't do anything if already running
        self.iter_count = 0
        if interval is not None:
            self.interval = interval
        if iterations is not None:
            self.max_iterations = iterations
        self._backend._vispy_start(self.interval)
        self._running = True
        self._first_emit_time = precision_time()
        self._last_emit_time = precision_time()
        self.events.start(type='timer_start')