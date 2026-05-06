def _initDMinfo(self):
        """Check files in /dev/mapper to initialize data structures for 
        mappings between device-mapper devices, minor device numbers, VGs 
        and LVs.
        
        """
        self._mapLVtuple2dm = {}
        self._mapLVname2dm = {}
        self._vgTree = {}
        if self._dmMajorNum is None:
            self._initBlockMajorMap()
        for file in os.listdir(devmapperDir):
            mobj = re.match('([a-zA-Z0-9+_.\-]*[a-zA-Z0-9+_.])-([a-zA-Z0-9+_.][a-zA-Z0-9+_.\-]*)$', file)
            if mobj:
                path = os.path.join(devmapperDir, file)
                (major, minor) = self._getDevMajorMinor(path)
                if major == self._dmMajorNum:
                    vg = mobj.group(1).replace('--', '-')
                    lv = mobj.group(2).replace('--', '-')
                    dmdev = "dm-%d" % minor
                    self._mapLVtuple2dm[(vg,lv)] = dmdev
                    self._mapLVname2dm[file] = dmdev
                    if not vg in self._vgTree:
                        self._vgTree[vg] = []
                    self._vgTree[vg].append(lv)