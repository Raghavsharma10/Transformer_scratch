def retrieveVals(self):
        """Retrieve values for graphs."""
        for iface in self._ifaceList:
            stats = self._ifaceStats.get(iface)
            graph_name = 'netiface_traffic_%s' % iface
            if self.hasGraph(graph_name):
                self.setGraphVal(graph_name, 'rx', stats.get('rxbytes') * 8)
                self.setGraphVal(graph_name, 'tx', stats.get('txbytes') * 8)
            graph_name = 'netiface_errors_%s' % iface
            if self.hasGraph(graph_name):
                for field in ('rxerrs', 'txerrs', 'rxframe', 'txcarrier',
                    'rxdrop', 'txdrop', 'rxfifo', 'txfifo'):
                    self.setGraphVal(graph_name, field, stats.get(field))