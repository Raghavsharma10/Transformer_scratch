def alarm_set(self, time, wake_with_radio=False):
        """
        set the alarm clock

        :param str time: time of the alarm (format: %H:%M:%S)
        :param bool wake_with_radio: if True, radio will be used for the alarm
                                     instead of beep sound
        """
        # TODO: check for correct time format
        log.debug("alarm => set...")
        params = {
            "enabled": True,
            "time": time,
            "wake_with_radio": wake_with_radio
        }
        self._app_exec("com.lametric.clock", "clock.alarm", params=params)