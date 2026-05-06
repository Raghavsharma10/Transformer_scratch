def _initDevClasses(self):
        """Sort block devices into lists depending on device class and 
        initialize device type map and partition map."""
        self._devClassTree = {}
        self._partitionTree = {}
        self._mapDevType = {}
        basedevs = []
        otherdevs = []
        if self._mapMajorDevclass is None:
            self._initBlockMajorMap()
        for dev in self._diskStats:
            stats = self._diskStats[dev]
            devclass = self._mapMajorDevclass.get(stats['major'])
            if devclass is not None:
                devdir = os.path.join(sysfsBlockdevDir, dev)
                if os.path.isdir(devdir):
                    if not self._devClassTree.has_key(devclass):
                        self._devClassTree[devclass] = []
                    self._devClassTree[devclass].append(dev)
                    self._mapDevType[dev] = devclass
                    basedevs.append(dev)
                else:
                    otherdevs.append(dev)
        basedevs.sort(key=len, reverse=True)
        otherdevs.sort(key=len, reverse=True)
        idx = 0
        for partdev in otherdevs:
            while len(basedevs[idx]) > partdev:
                idx += 1 
            for dev in basedevs[idx:]:
                if re.match("%s(\d+|p\d+)$" % dev, partdev):
                    if not self._partitionTree.has_key(dev):
                        self._partitionTree[dev] = []
                    self._partitionTree[dev].append(partdev)
                    self._mapDevType[partdev] = 'part'