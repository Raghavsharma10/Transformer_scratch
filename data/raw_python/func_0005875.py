def set_metrics_threshold(self, name, value, check_interval=None, reset_to=None, alarm=None, alarm_message=None):
        """Sets metric threshold parameters.

        :param str|unicode name: Metric name.

        :param int value: Threshold value.

        :param int reset_to: Reset value to when threshold is reached.

        :param int check_interval: Threshold check interval in seconds.

        :param str|unicode|AlarmType alarm: Alarm to trigger when threshold is reached.

        :param str|unicode alarm_message: Message to pass to alarm. If not set metrics name is passed.

        """
        if alarm is not None and isinstance(alarm, AlarmType):
            self._section.alarms.register_alarm(alarm)
            alarm = alarm.alias

        value = KeyValue(
            locals(),
            aliases={
                'name': 'key',
                'reset_to': 'reset',
                'check_interval': 'rate',
                'alarm_message': 'msg',
            },
        )

        self._set('metric-threshold', value, multi=True)

        return self._section