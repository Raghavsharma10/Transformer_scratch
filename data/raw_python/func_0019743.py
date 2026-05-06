def ping(self):
        """Ping Redis Server and return Round-Trip-Time in seconds.
        
        @return: Round-trip-time in seconds as float.
        
        """
        start = time.time()
        self._conn.ping()
        return (time.time() - start)