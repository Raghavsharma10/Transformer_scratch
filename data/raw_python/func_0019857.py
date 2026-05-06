def retrieveVals(self):
        """Retrieve values for graphs."""
        net_info = NetstatInfo()
        if self.hasGraph('netstat_conn_status'):
            stats = net_info.getTCPportConnStatus(include_listen=True)
            for fname in ('listen', 'established', 'syn_sent', 'syn_recv',
                          'fin_wait1', 'fin_wait2', 'time_wait', 
                          'close','close_wait', 'last_ack', 'closing', 
                          'unknown',):
                self.setGraphVal('netstat_conn_status', fname, 
                                 stats.get(fname,0))
        if self.hasGraph('netstat_conn_server'):
            stats = net_info.getTCPportConnCount(localport=self._port_list)
            for srv in self._srv_list:
                numconn = 0
                for port in self._srv_dict[srv]:
                    numconn += stats.get(port, 0)
                self.setGraphVal('netstat_conn_server', srv, numconn)