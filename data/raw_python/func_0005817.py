def set_throttle_params(self, level=None, level_max=None):
        """Throttling options.

        * http://uwsgi-docs.readthedocs.io/en/latest/Emperor.html#throttling
        * http://uwsgi-docs.readthedocs.io/en/latest/Emperor.html#loyalty

        :param int level: Set throttling level (in milliseconds) for bad behaving vassals. Default: 1000.

        :param int level_max: Set maximum throttling level (in milliseconds)
            for bad behaving vassals. Default: 3 minutes.

        """
        self._set('emperor-throttle', level)
        self._set('emperor-max-throttle', level_max)

        return self._section