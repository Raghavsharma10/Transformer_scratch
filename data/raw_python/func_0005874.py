def set_metrics_params(self, enable=None, store_dir=None, restore=None, no_cores=None):
        """Sets basic Metrics subsystem params.

        uWSGI metrics subsystem allows you to manage "numbers" from your apps.

        When enabled, the subsystem configures a vast amount of metrics
        (like requests per-core, memory usage, etc) but, in addition to this,
        you can configure your own metrics, such as the number of active users or, say,
        hits of a particular URL, as well as the memory consumption of your app or the whole server.

        * http://uwsgi.readthedocs.io/en/latest/Metrics.html
        * SNMP Integration - http://uwsgi.readthedocs.io/en/latest/Metrics.html#snmp-integration

        :param bool enable: Enables the subsystem.

        :param str|unicode store_dir: Directory to store metrics.
            The metrics subsystem can expose all of its metrics in the form
            of text files in a directory. The content of each file is the value
            of the metric (updated in real time).

            .. note:: Placeholders can be used to build paths, e.g.: {project_runtime_dir}/metrics/
              See ``Section.project_name`` and ``Section.runtime_dir``.

        :param bool restore: Restore previous metrics from ``store_dir``.
            When you restart a uWSGI instance, all of its metrics are reset.
            Use the option to force the metric subsystem to read-back the values
            from the metric directory before starting to collect values.

        :param bool no_cores: Disable generation of cores-related metrics.

        """
        self._set('enable-metrics', enable, cast=bool)
        self._set('metrics-dir', self._section.replace_placeholders(store_dir))
        self._set('metrics-dir-restore', restore, cast=bool)
        self._set('metrics-no-cores', no_cores, cast=bool)

        return self._section