def _configDevActive(self, namestr, titlestr, devlist):
        """Generate configuration for I/O Queue Length.
        
        @param namestr:  Field name component indicating device type.
        @param titlestr: Title component indicating device type.
        @param devlist:  List of devices.
        
        """
        name = 'diskio_%s_active' % namestr
        if self.graphEnabled(name):
            graph = MuninGraph('Disk I/O - %s - Queue Length' % titlestr, 
                self._category,
                info='Disk I/O - Number  of I/O Operations in Progress for every %s.'
                     % titlestr,
                args='--base 1000 --lower-limit 0', printf='%6.1lf',
                autoFixNames = True)
            for dev in devlist:
                graph.addField(dev, 
                               fixLabel(dev, maxLabelLenGraphSimple, 
                                        repl = '..', truncend=False,
                                        delim = self._labelDelim.get(namestr)), 
                               draw='AREASTACK', type='GAUGE', info=dev)
            self.appendGraph(name, graph)