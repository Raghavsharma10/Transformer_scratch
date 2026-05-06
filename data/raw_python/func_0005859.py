def alarm_on_segfault(self, alarm):
        """Raise the specified alarm when the segmentation fault handler is executed.

        Sends a backtrace.

        :param AlarmType|list[AlarmType] alarm: Alarm.
        """
        self.register_alarm(alarm)

        for alarm in listify(alarm):
            self._set('alarm-segfault', alarm.alias, multi=True)

        return self._section