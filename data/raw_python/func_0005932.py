def get_message(cls, signals=True, farms=False, buffer_size=65536, timeout=-1):
        """Block until a mule message is received and return it.

        This can be called from multiple threads in the same programmed mule.

        :param bool signals: Whether to manage signals.

        :param bool farms: Whether to manage farms.

        :param int buffer_size:

        :param int timeout: Seconds.

        :rtype: str|unicode

        :raises ValueError: If not in a mule.
        """
        return decode(uwsgi.mule_get_msg(signals, farms, buffer_size, timeout))