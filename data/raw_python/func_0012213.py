def set_screensaver(
        self, mode, is_mode_enabled, start_time=None, end_time=None,
        is_screensaver_enabled=True
    ):
        """
        set the display's screensaver mode

        :param str mode: mode of the screensaver
                         [when_dark, time_based]
        :param bool is_mode_enabled: specifies if mode is enabled or disabled
        :param str start_time: start time, only used in time_based mode
                               (format: %H:%M:%S)
        :param str end_time: end time, only used in time_based mode
                             (format: %H:%M:%S)
        :param bool is_screensaver_enabled: is overall screensaver turned on
                                            overrules mode specific settings
        """
        assert(mode in ("when_dark", "time_based"))

        log.debug("setting screensaver to '{}'...".format(mode))

        cmd, url = DEVICE_URLS["set_display"]
        json_data = {
            "screensaver": {
                "enabled": is_screensaver_enabled,
                "mode": mode,
                "mode_params": {
                    "enabled": is_mode_enabled
                },
            }
        }
        if mode == "time_based":
            # TODO: add time checks
            assert((start_time is not None) and (end_time is not None))
            json_data["screensaver"]["mode_params"]["start_time"] = start_time
            json_data["screensaver"]["mode_params"]["end_time"] = end_time

        return self._exec(cmd, url, json_data=json_data)