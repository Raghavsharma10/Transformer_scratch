def set_basic_params(self, msg_size=None, cheap=None, anti_loop_timeout=None):
        """
        :param int msg_size: Set the max size of an alarm message in bytes. Default: 8192.

        :param bool cheap: Use main alarm thread rather than create dedicated
            threads for curl-based alarms

        :param int anti_loop_timeout: Tune the anti-loop alarm system. Default: 3 seconds.

        """
        self._set('alarm-msg-size', msg_size)
        self._set('alarm-cheap', cheap, cast=bool)
        self._set('alarm-freq', anti_loop_timeout)

        return self._section