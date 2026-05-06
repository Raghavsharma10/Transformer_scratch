def restart(self):
        """
        Tells the HAProxy control object to restart the process.

        If it's been fewer than `restart_interval` seconds since the previous
        restart, it will wait until the interval has passed.  This staves off
        situations where the process is constantly restarting, as it is
        possible to drop packets for a short interval while doing so.
        """
        delay = (self.last_restart - time.time()) + self.restart_interval

        if delay > 0:
            time.sleep(delay)

        self.control.restart()

        self.last_restart = time.time()
        self.restart_required = False