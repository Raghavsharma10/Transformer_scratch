def _configDevRequests(self, namestr, titlestr, devlist):
        """Generate configuration for I/O Request stats.
        
        @param namestr:  Field name component indicating device type.
        @param titlestr: Title component indicating device type.
        @param devlist:  List of devices.
        
        """
        name = 'diskio_%s_requests' % namestr
        if self.graphEnabled(name):
            graph = MuninGraph('Disk I/O - %s - Requests' % titlestr, self._category,
                info='Disk I/O - %s Throughput, Read / write requests per second.' 
                     % titlestr,
                args='--base 1000 --lower-limit 0',
                vlabel='reqs/sec read (-) / write (+)', printf='%6.1lf',
                autoFixNames = True)
            for dev in devlist:
                graph.addField(dev + '_read',
                               fixLabel(dev, maxLabelLenGraphDual, 
                                        repl = '..', truncend=False,
                                        delim = self._labelDelim.get(namestr)), 
                               draw='LINE2', type='DERIVE', min=0, graph=False)
                graph.addField(dev + '_write',
                               fixLabel(dev, maxLabelLenGraphDual, 
                                        repl = '..', truncend=False,
                                        delim = self._labelDelim.get(namestr)),
                               draw='LINE2', type='DERIVE', min=0, 
                               negative=(dev + '_read'),info=dev)
            self.appendGraph(name, graph)