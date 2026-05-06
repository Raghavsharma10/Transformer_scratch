def _is_running(self, tries=10):
        """
        Return if the server is running according to pg_ctl.
        """
        # We can't possibly be running if our base_pathname isn't defined.
        if not self.base_pathname:
            return False

        if tries < 1:
            raise ValueError('tries must be > 0')

        cmd = [
            PostgresFinder.find_root() / 'pg_ctl',
            'status',
            '-D',
            self.base_pathname,
        ]
        votes = 0
        while abs(votes) < tries:
            time.sleep(0.1)
            running = (subprocess.call(cmd, stdout=DEV_NULL) == 0)
            if running and votes >= 0:
                votes += 1
            elif not running and votes <= 0:
                votes -= 1
            else:
                votes = 0

        return votes > 0