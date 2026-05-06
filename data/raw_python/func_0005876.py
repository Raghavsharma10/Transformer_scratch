def set_stats_params(
            self, address=None, enable_http=None,
            minify=None, no_cores=None, no_metrics=None, push_interval=None):
        """Enables stats server on the specified address.

        * http://uwsgi.readthedocs.io/en/latest/StatsServer.html

        :param str|unicode address: Address/socket to make stats available on.

            Examples:
                * 127.0.0.1:1717
                * /tmp/statsock
                * :5050

        :param bool enable_http: Server stats over HTTP.
            Prefixes stats server json output with http headers.

        :param bool minify: Minify statistics json output.

        :param bool no_cores: Disable generation of cores-related stats.

        :param bool no_metrics: Do not include metrics in stats output.

        :param int push_interval: Set the default frequency of stats pushers in seconds/

        """
        self._set('stats-server', address)
        self._set('stats-http', enable_http, cast=bool)
        self._set('stats-minified', minify, cast=bool)
        self._set('stats-no-cores', no_cores, cast=bool)
        self._set('stats-no-metrics', no_metrics, cast=bool)
        self._set('stats-pusher-default-freq', push_interval)

        return self._section