def _initSwapInfo(self):
        """Initialize swap partition to device mappings."""
        self._swapList = []
        sysinfo = SystemInfo()
        for (swap,attrs) in sysinfo.getSwapStats().iteritems():
            if attrs['type'] == 'partition':
                dev = self._getUniqueDev(swap)
                if dev is not None:
                    self._swapList.append(dev)