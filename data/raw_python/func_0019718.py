def _initDiskStats(self):
        """Parse and initialize block device I/O stats in /proc/diskstats."""
        self._diskStats = {}
        self._mapMajorMinor2dev = {}
        try:
            fp = open(diskStatsFile, 'r')
            data = fp.read()
            fp.close()
        except:
            raise IOError('Failed reading disk stats from file: %s'
                          % diskStatsFile)
        for line in data.splitlines():
            cols = line.split()
            dev = cols.pop(2)
            if len(cols) == 13:
                self._diskStats[dev] = dict(zip(
                    ('major', 'minor',
                     'rios', 'rmerges', 'rsect', 'rticks',
                     'wios', 'wmerges', 'wsect', 'wticks',
                     'ios_active', 'totticks', 'rqticks'),
                    [int(x) for x in cols]))
            elif len(cols) == 6:
                self._diskStats[dev] = dict(zip(
                    ('major', 'minor',
                     'rios', 'rsect',
                     'wios', 'wsect'),
                    [int(x) for x in cols]))
            else:
                continue
            self._diskStats[dev]['rbytes'] = (
                self._diskStats[dev]['rsect'] * sectorSize)
            self._diskStats[dev]['wbytes'] = (
                self._diskStats[dev]['wsect'] * sectorSize)
            self._mapMajorMinor2dev[(int(cols[0]), int(cols[1]))] = dev