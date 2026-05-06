def _getUniqueDev(self, devpath):
        """Return unique device for any block device path.
        
        @param devpath: Full path for block device.
        @return:        Unique device string without the /dev prefix.
        
        """
        realpath = os.path.realpath(devpath)
        mobj = re.match('\/dev\/(.*)$', realpath)
        if mobj:
            dev = mobj.group(1)
            if dev in self._diskStats:
                return dev
            else:
                try:
                    (major, minor) = self._getDevMajorMinor(realpath)
                except:
                    return None
                return self._mapMajorMinor2dev.get((major, minor))
        return None