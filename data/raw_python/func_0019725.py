def retrieveVals(self):
        """Retrieve values for graphs."""
        ntpinfo = NTPinfo()
        stats = ntpinfo.getHostOffset(self._remoteHost)
        if stats:
            graph_name = 'ntp_host_stratum_%s' % self._remoteHost
            if self.hasGraph(graph_name):
                self.setGraphVal(graph_name, 'stratum', stats.get('stratum'))
            graph_name = 'ntp_host_stat_%s' % self._remoteHost
            if self.hasGraph(graph_name):
                self.setGraphVal(graph_name, 'offset', stats.get('offset'))
                self.setGraphVal(graph_name, 'delay', stats.get('delay'))