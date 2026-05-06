def cmd_reload(self, force=False, workers_only=False, workers_chain=False):
        """Reloads uWSGI master process, workers.

        :param bool force: Use forced (brutal) reload instead of a graceful one.
        :param bool workers_only: Reload only workers.
        :param bool workers_chain: Run chained workers reload (one after another,
            instead of destroying all of them in bulk).

        """
        if workers_chain:
            return self.send_command(b'c')

        if workers_only:
            return self.send_command(b'R' if force else b'r')

        return self.send_command(b'R' if force else b'r')