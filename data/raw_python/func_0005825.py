def set_reload_params(
            self, min_lifetime=None, max_lifetime=None,
            max_requests=None, max_requests_delta=None,
            max_addr_space=None, max_rss=None, max_uss=None, max_pss=None,
            max_addr_space_forced=None, max_rss_forced=None, watch_interval_forced=None,
            mercy=None):
        """Sets workers reload parameters.

        :param int min_lifetime: A worker cannot be destroyed/reloaded unless it has been alive
            for N seconds (default 60). This is an anti-fork-bomb measure.
            Since 1.9

        :param int max_lifetime: Reload workers after this many seconds. Disabled by default.
            Since 1.9

        :param int max_requests: Reload workers after the specified amount of managed
            requests (avoid memory leaks).
            When a worker reaches this number of requests it will get recycled (killed and restarted).
            You can use this option to "dumb fight" memory leaks.

            Also take a look at the ``reload-on-as`` and ``reload-on-rss`` options
            as they are more useful for memory leaks.

            .. warning:: The default min-worker-lifetime 60 seconds takes priority over `max-requests`.

            Do not use with benchmarking as you'll get stalls
            such as `worker respawning too fast !!! i have to sleep a bit (2 seconds)...`

        :param int max_requests_delta: Add (worker_id * delta) to the max_requests value of each worker.

        :param int max_addr_space: Reload a worker if its address space usage is higher
            than the specified value in megabytes.

        :param int max_rss: Reload a worker if its physical unshared memory (resident set size) is higher
            than the specified value (in megabytes).

        :param int max_uss: Reload a worker if Unique Set Size is higher
            than the specified value in megabytes.

            .. note:: Linux only.

        :param int max_pss: Reload a worker if Proportional Set Size is higher
            than the specified value in megabytes.

            .. note:: Linux only.

        :param int max_addr_space_forced: Force the master to reload a worker if its address space is higher
            than specified megabytes (in megabytes).

        :param int max_rss_forced: Force the master to reload a worker
            if its resident set size memory is higher than specified in megabytes.

        :param int watch_interval_forced: The memory collector [per-worker] thread memeory watch
            interval (seconds) used for forced reloads. Default: 3.

        :param int mercy: Set the maximum time (in seconds) a worker can take
            before reload/shutdown. Default: 60.

        """
        self._set('max-requests', max_requests)
        self._set('max-requests-delta', max_requests_delta)

        self._set('min-worker-lifetime', min_lifetime)
        self._set('max-worker-lifetime', max_lifetime)

        self._set('reload-on-as', max_addr_space)
        self._set('reload-on-rss', max_rss)
        self._set('reload-on-uss', max_uss)
        self._set('reload-on-pss', max_pss)

        self._set('evil-reload-on-as', max_addr_space_forced)
        self._set('evil-reload-on-rss', max_rss_forced)
        self._set('mem-collector-freq', watch_interval_forced)

        self._set('worker-reload-mercy', mercy)

        return self._section