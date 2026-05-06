def set_volume(self, volume=50):
        """
        allows to change the volume

        :param int volume: volume to be set for the current device
                           [0..100] (default: 50)
        """
        assert(volume in range(101))

        log.debug("setting volume...")

        cmd, url = DEVICE_URLS["set_volume"]
        json_data = {
            "volume": volume,
        }
        return self._exec(cmd, url, json_data=json_data)