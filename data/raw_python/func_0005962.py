def set_basic_params(self, no_expire=None, expire_scan_interval=None, report_freed=None):
        """
        :param bool no_expire: Disable auto sweep of expired items.
            Since uWSGI 1.2, cache item expiration is managed by a thread in the master process,
            to reduce the risk of deadlock. This thread can be disabled
            (making item expiry a no-op) with the this option.

        :param int expire_scan_interval: Set the frequency (in seconds) of cache sweeper scans. Default: 3.

        :param bool report_freed: Constantly report the cache item freed by the sweeper.

            .. warning:: Use only for debug.

        """
        self._set('cache-no-expire', no_expire, cast=bool)
        self._set('cache-report-freed-items', report_freed, cast=bool)
        self._set('cache-expire-freq', expire_scan_interval)

        return self._section