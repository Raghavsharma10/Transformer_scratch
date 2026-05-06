def retrieveVals(self):
        """Retrieve values for graphs."""
        ntpinfo = NTPinfo()
        ntpstats = ntpinfo.getHostOffsets(self._remoteHosts)
        if ntpstats:
            for host in self._remoteHosts:
                hostkey = re.sub('\.', '_', host)
                hoststats = ntpstats.get(host)
                if hoststats:
                    if self.hasGraph('ntp_host_stratums'):
                        self.setGraphVal('ntp_host_stratums', hostkey, 
                                         hoststats.get('stratum'))
                    if self.hasGraph('ntp_host_offsets'):
                        self.setGraphVal('ntp_host_offsets', hostkey, 
                                         hoststats.get('offset'))
                    if self.hasGraph('ntp_host_delays'):
                        self.setGraphVal('ntp_host_delays', hostkey, 
                                         hoststats.get('delay'))