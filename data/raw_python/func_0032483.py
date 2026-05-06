def scan_field(self, measure, field, rate, mode='persistent', delay=1):
        """Performs a field scan.

        Measures until the target field is reached.

        :param measure: A callable called repeatedly until stability at the
            target field is reached.
        :param field: The target field in Oersted.

            .. note:: The conversion is 1 Oe = 0.1 mT.

        :param rate: The field rate in Oersted per minute.
        :param mode: The state of the magnet at the end of the charging
            process, either 'persistent' or 'driven'.
        :param delay: The time delay between each call to measure in seconds.

        :raises TypeError: if measure parameter is not callable.

        """
        if not hasattr(measure, '__call__'):
            raise TypeError('measure parameter not callable.')
        self.set_field(field, rate, approach='linear', mode=mode, wait_for_stability=False)
        if self.system_status['magnet'].startswith('persist'):
            # The persistent switch takes some time to open. While it's opening,
            # the status does not change.
            switch_heat_time = datetime.timedelta(seconds=self.magnet_config[5])
            start = datetime.datetime.now()
            while True:
                now = datetime.datetime.now()
                if now - start > switch_heat_time:
                    break
                measure()
                time.sleep(delay)
        while True:
            status = self.system_status['magnet']
            if status in ('persistent, stable', 'driven, stable'):
                break
            measure()
            time.sleep(delay)