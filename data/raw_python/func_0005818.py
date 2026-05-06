def set_tolerance_params(self, for_heartbeat=None, for_cursed_vassals=None):
        """Various tolerance options.

        :param int for_heartbeat: Set the Emperor tolerance about heartbeats.

            * http://uwsgi-docs.readthedocs.io/en/latest/Emperor.html#heartbeat-system

        :param int for_cursed_vassals: Set the Emperor tolerance about cursed vassals.

            * http://uwsgi-docs.readthedocs.io/en/latest/Emperor.html#blacklist-system

        """
        self._set('emperor-required-heartbeat', for_heartbeat)
        self._set('emperor-curse-tolerance', for_cursed_vassals)

        return self._section