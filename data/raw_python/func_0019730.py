def _retrieve(self, map):
        """Query Squid Proxy Server Manager Interface for stats.
        
        @param map: Statistics map name.
        @return:    Dictionary of query results.
        
        """
        self._conn.request('GET', "cache_object://%s/%s" % (self._host, map), 
                           None, self._httpHeaders)
        rp = self._conn.getresponse()
        if rp.status == 200:
            data = rp.read()
            return data
        else:
            raise Exception("Retrieval of stats from Squid Proxy Server"
                            "on host %s and port %s failed.\n"
                            "HTTP - Status: %s    Reason: %s" 
                            % (self._host, self._port, rp.status, rp.reason))