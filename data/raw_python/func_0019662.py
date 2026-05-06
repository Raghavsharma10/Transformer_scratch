def retrieveVals(self):
        """Retrieve values for graphs."""   
        nginxInfo = NginxInfo(self._host, self._port,
                                self._user, self._password, 
                                self._statuspath, self._ssl)
        stats = nginxInfo.getServerStats()
        if stats:
            if self.hasGraph('nginx_activeconn'):
                self.setGraphVal('nginx_activeconn', 'proc', stats['writing'])
                self.setGraphVal('nginx_activeconn', 'read', stats['reading'])
                self.setGraphVal('nginx_activeconn', 'wait', stats['waiting'])
                self.setGraphVal('nginx_activeconn', 'total', 
                                 stats['connections'])
            if self.hasGraph('nginx_connections'):
                self.setGraphVal('nginx_connections', 'handled', stats['handled'])
                self.setGraphVal('nginx_connections', 'nothandled', 
                                 stats['accepts'] - stats['handled'])
            if self.hasGraph('nginx_requests'):
                self.setGraphVal('nginx_requests', 'requests', stats['requests'])
            if self.hasGraph('nginx_requestsperconn'):
                curr_stats = (stats['handled'], stats['requests'])
                hist_stats = self.restoreState()
                if hist_stats:
                    prev_stats = hist_stats[0]
                else:
                    hist_stats = []
                    prev_stats = (0,0)
                conns = max(curr_stats[0] - prev_stats[0], 0)
                reqs = max(curr_stats[1] - prev_stats[1], 0)
                if conns > 0:
                    self.setGraphVal('nginx_requestsperconn', 'requests',
                                     float(reqs) / float(conns))
                else:
                    self.setGraphVal('nginx_requestsperconn', 'requests', 0)
                hist_stats.append(curr_stats)
                self.saveState(hist_stats[-self._numSamples:])