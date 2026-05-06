def autoconf(self):
        """Implements Munin Plugin Auto-Configuration Option.
        
        @return: True if plugin can be  auto-configured, False otherwise.
                 
        """
        ntpinfo = NTPinfo()
        ntpstats = ntpinfo.getHostOffsets(self._remoteHosts)
        return len(ntpstats) > 0