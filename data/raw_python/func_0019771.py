def autoconf(self):
        """Implements Munin Plugin Auto-Configuration Option.
        
        @return: True if plugin can be  auto-configured, False otherwise.
                 
        """
        serverInfo = MemcachedInfo(self._host,  self._port, self._socket_file)
        return (serverInfo is not None)