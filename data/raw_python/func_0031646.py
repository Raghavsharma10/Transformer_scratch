def firmware_version(self):
        """
        Provides information on the connected Pebble, including its firmware version, language, capabilities, etc.

        .. note:
           This is a blocking call if :meth:`fetch_watch_info` has not yet been called, which could lead to deadlock
           if called in an endpoint callback.

        :rtype: .WatchVersionResponse
        """
        version = self.watch_info.running.version_tag[1:]
        parts = version.split('-', 1)
        points = [int(x) for x in parts[0].split('.')]
        while len(points) < 3:
            points.append(0)
        if len(parts) == 2:
            suffix = parts[1]
        else:
            suffix = ''
        return FirmwareVersion(*(points + [suffix]))