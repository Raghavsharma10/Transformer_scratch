def countdown_set(self, duration, start_now):
        """
        set the countdown

        :param str duration:
        :param str start_now:
        """
        log.debug("countdown => set...")
        params = {'duration': duration, 'start_now': start_now}
        self._app_exec(
            "com.lametric.countdown", "countdown.configure", params
        )