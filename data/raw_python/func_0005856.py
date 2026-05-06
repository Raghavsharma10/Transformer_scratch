def register_alarm(self, alarm):
        """Register (create) an alarm.

        :param AlarmType|list[AlarmType] alarm: Alarm.

        """
        for alarm in listify(alarm):
            if alarm not in self._alarms:
                self._set('alarm', alarm, multi=True)
                self._alarms.append(alarm)

        return self._section