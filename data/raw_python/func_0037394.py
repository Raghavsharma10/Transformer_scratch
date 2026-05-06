def ready(self):
        """
        Assumes postgres now talks to pg_ctl, but might not yet be listening
        or connections from psql.  Test that psql is able to connect, as
        it occasionally takes 5-10 seconds for postgresql to start listening.
        """
        cmd = self._psql_cmd()
        for i in range(50, -1, -1):
            res = subprocess.call(
                cmd, stdin=DEV_NULL, stdout=DEV_NULL,
                stderr=DEV_NULL)
            if res == 0:
                break
            time.sleep(0.2)
        return i != 0