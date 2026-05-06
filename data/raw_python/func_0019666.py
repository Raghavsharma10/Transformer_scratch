def autoconf(self):
        """Implements Munin Plugin Auto-Configuration Option.
        
        @return: True if plugin can be  auto-configured, False otherwise.
                 
        """
        apcinfo = APCinfo(self._host, self._port, self._user, self._password, 
                          self._monpath, self._ssl)
        return apcinfo is not None