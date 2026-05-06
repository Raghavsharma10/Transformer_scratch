def stop(self):
        """Stop the timer."""
        self._backend._vispy_stop()
        self._running = False
        self.events.stop(type='timer_stop')