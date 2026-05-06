def alarm_on_log(self, alarm, matcher, skip=False):
        """Raise (or skip) the specified alarm when a log line matches the specified regexp.

        :param AlarmType|list[AlarmType] alarm: Alarm.

        :param str|unicode matcher: Regular expression to match log line.

        :param bool skip:

        """
        self.register_alarm(alarm)

        value = '%s %s' % (
            ','.join(map(attrgetter('alias'), listify(alarm))),
            matcher)

        self._set('not-alarm-log' if skip else 'alarm-log', value)

        return self._section