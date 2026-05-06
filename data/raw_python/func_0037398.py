def stop(self):
        """Stop this DMBS daemon.  If it's not currently running, do nothing.

        Don't return until it's terminated.
        """
        log.info('Stopping PostgreSQL at %s:%s', self.host, self.port)
        if self._is_running():
            cmd = [
                PostgresFinder.find_root() / 'pg_ctl',
                'stop',
                '-D', self.base_pathname,
                '-m', 'fast',
            ]
            subprocess.check_call(cmd)
            # pg_ctl isn't reliable if it's called at certain critical times
            if self.pid:
                os.kill(self.pid, signal.SIGTERM)
        # Can't use wait() because the server might not be our child
        while self._is_running():
            time.sleep(0.1)