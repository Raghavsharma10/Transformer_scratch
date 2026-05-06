def at(self, cmd, *args, **kwargs):
        """Wrapper for the low level at commands.

        This method takes care that the sequence number is increased after each
        at command and the watchdog timer is started to make sure the drone
        receives a command at least every second.
        """
        with self.lock:
            self.com_watchdog_timer.cancel()
            cmd(self.host, self.sequence, *args, **kwargs)
            self.sequence += 1
            self.com_watchdog_timer = threading.Timer(self.timer, self.commwdg)
            self.com_watchdog_timer.start()