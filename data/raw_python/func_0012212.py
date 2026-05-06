def set_display(self, brightness=100, brightness_mode="auto"):
        """
        allows to modify display state (change brightness)

        :param int brightness: display brightness [0, 100] (default: 100)
        :param str brightness_mode: the brightness mode of the display
                                    [auto, manual] (default: auto)
        """
        assert(brightness_mode in ("auto", "manual"))
        assert(brightness in range(101))

        log.debug("setting display information...")

        cmd, url = DEVICE_URLS["set_display"]
        json_data = {
            "brightness_mode": brightness_mode,
            "brightness": brightness
        }

        return self._exec(cmd, url, json_data=json_data)