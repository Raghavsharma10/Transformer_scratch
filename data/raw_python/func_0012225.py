def alarm_disable(self):
        """
        disable the alarm
        """
        log.debug("alarm => disable...")
        params = {"enabled": False}
        self._app_exec("com.lametric.clock", "clock.alarm", params=params)