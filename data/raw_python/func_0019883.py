def retrieveVals(self):
        """Retrieve values for graphs."""
        for iface in self._ifaceList:
            if self._reqIfaceList is None or iface in self._reqIfaceList:
                if (self.graphEnabled('wanpipe_traffic') 
                    or self.graphEnabled('wanpipe_errors')):
                    stats = self._ifaceStats.get(iface)
                    if stats:
                        graph_name = 'wanpipe_traffic_%s' % iface
                        if self.hasGraph(graph_name):
                            for field in ('rxpackets', 'txpackets'):
                                self.setGraphVal(graph_name, field, 
                                                 stats.get(field))
                        graph_name = 'wanpipe_errors_%s' % iface
                        if self.hasGraph(graph_name):
                            for field in ('rxerrs', 'txerrs', 'rxframe', 'txcarrier',
                                'rxdrop', 'txdrop', 'rxfifo', 'txfifo'):
                                self.setGraphVal(graph_name, field, 
                                                 stats.get(field))
                if (self.graphEnabled('wanpipe_pri_errors') 
                    or self.graphEnabled('wanpipe_rxlevel')):
                    try:
                        stats = self._wanpipeInfo.getPRIstats(iface)
                    except:
                        stats = None
                    if stats:
                        graph_name = 'wanpipe_pri_errors_%s' % iface
                        if self.hasGraph(graph_name):
                            for field in ('linecodeviolation', 
                                          'farendblockerrors',
                                          'crc4errors', 'faserrors'):
                                self.setGraphVal(graph_name, field, 
                                                 stats.get(field))
                        if self.hasGraph('wanpipe_rxlevel'):
                            self.setGraphVal('wanpipe_pri_rxlevel', 
                                             iface, stats.get('rxlevel'))