def autoconf(self):
        """Implements Munin Plugin Auto-Configuration Option.
        
        @return: True if plugin can be  auto-configured, False otherwise.
                 
        """
        fs = FSinfo(self._fshost, self._fsport, self._fspass)
        return fs is not None