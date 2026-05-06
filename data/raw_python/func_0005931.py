def set(self, value, mode=None):
        """Sets metric value.

        :param int|long value: New value.

        :param str|unicode mode: Update mode.

            * None - Unconditional update.
            * max - Sets metric value if it is greater that the current one.
            * min - Sets metric value if it is less that the current one.

        :rtype: bool

        """
        if mode == 'max':
            func = uwsgi.metric_set_max

        elif mode == 'min':
            func = uwsgi.metric_set_min

        else:
            func = uwsgi.metric_set

        return func(self.name, value)