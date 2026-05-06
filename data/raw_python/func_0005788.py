def set_basic_params(
            self, check_interval_busy=None,
            busy_max=None, busy_min=None,
            idle_cycles_max=None, idle_cycles_penalty=None,
            verbose=None):
        """
        :param int check_interval_busy: Interval (sec) to check worker busyness.

        :param int busy_max: Maximum busyness (percents). Every time the calculated busyness
            is higher than this value, uWSGI will spawn new workers. Default: 50.

        :param int busy_min: Minimum busyness (percents). If busyness is below this value,
            the app is considered in an "idle cycle" and uWSGI will start counting them.
            Once we reach needed number of idle cycles uWSGI will kill one worker. Default: 25.

        :param int idle_cycles_max: This option tells uWSGI how many idle cycles are allowed
            before stopping a worker.

        :param int idle_cycles_penalty: Number of idle cycles to add to ``idle_cycles_max``
            in case worker spawned too early. Default is 1.

        :param bool verbose: Enables debug logs for this algo.

        """
        self._set('cheaper-overload', check_interval_busy)
        self._set('cheaper-busyness-max', busy_max)
        self._set('cheaper-busyness-min', busy_min)
        self._set('cheaper-busyness-multiplier', idle_cycles_max)
        self._set('cheaper-busyness-penalty', idle_cycles_penalty)
        self._set('cheaper-busyness-verbose', verbose, cast=bool)

        return self._section