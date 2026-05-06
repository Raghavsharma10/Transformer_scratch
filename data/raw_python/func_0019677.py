def _fetchDevAll(self, namestr, devlist, statsfunc):
        """Initialize I/O stats for devices.
        
        @param namestr:   Field name component indicating device type.
        @param devlist:   List of devices.
        @param statsfunc: Function for retrieving stats for device.
        
        """
        for dev in devlist:
            stats = statsfunc(dev)
            name = 'diskio_%s_requests' % namestr
            if self.hasGraph(name):
                self.setGraphVal(name, dev + '_read', stats['rios'])
                self.setGraphVal(name, dev + '_write', stats['wios'])
            name = 'diskio_%s_bytes' % namestr
            if self.hasGraph(name):
                self.setGraphVal(name, dev + '_read', stats['rbytes'])
                self.setGraphVal(name, dev + '_write', stats['wbytes'])
            name = 'diskio_%s_active' % namestr
            if self.hasGraph(name):
                self.setGraphVal(name, dev, stats['ios_active'])