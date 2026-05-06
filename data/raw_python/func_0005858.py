def alarm_on_fd_ready(self, alarm, fd, message, byte_count=None):
        """Triggers the alarm when the specified file descriptor is ready for read.

        This is really useful for integration with the Linux eventfd() facility.
        Pretty low-level and the basis of most of the alarm plugins.

        * http://uwsgi-docs.readthedocs.io/en/latest/Changelog-1.9.7.html#alarm-fd

        :param AlarmType|list[AlarmType] alarm: Alarm.

        :param str|unicode fd: File descriptor.

        :param str|unicode message: Message to send.

        :param int byte_count: Files to read. Default: 1 byte.

            .. note:: For ``eventfd`` set 8.

        """
        self.register_alarm(alarm)

        value = fd

        if byte_count:
            value += ':%s' % byte_count

        value += ' %s' % message

        for alarm in listify(alarm):
            self._set('alarm-fd', '%s %s' % (alarm.alias, value), multi=True)

        return self._section