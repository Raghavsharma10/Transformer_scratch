def _initBlockMajorMap(self):
        """Parses /proc/devices to initialize device class - major number map
        for block devices.
        
        """
        self._mapMajorDevclass = {}
        try:
            fp = open(devicesFile, 'r')
            data = fp.read()
            fp.close()
        except:
            raise IOError('Failed reading device information from file: %s'
                          % devicesFile)
        skip = True
        for line in data.splitlines():
            if skip:
                if re.match('block.*:', line, re.IGNORECASE):
                    skip = False
            else:
                mobj = re.match('\s*(\d+)\s+([\w\-]+)$', line)
                if mobj:
                    major = int(mobj.group(1))
                    devtype = mobj.group(2)
                    self._mapMajorDevclass[major] = devtype
                    if devtype == 'device-mapper':
                        self._dmMajorNum = major