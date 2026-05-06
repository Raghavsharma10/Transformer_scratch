def autoconf(self):
        """Implements Munin Plugin Auto-Configuration Option.
        
        @return: True if plugin can be  auto-configured, False otherwise.
                 
        """
        apacheInfo = ApacheInfo(self._host, self._port,
                                self._user, self._password, 
                                self._statuspath, self._ssl)
        return apacheInfo is not None