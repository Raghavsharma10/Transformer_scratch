def set_field(self, field, rate, approach='linear', mode='persistent',
                  wait_for_stability=True, delay=1):
        """Sets the magnetic field.

        :param field: The target field in Oersted.

            .. note:: The conversion is 1 Oe = 0.1 mT.

        :param rate: The field rate in Oersted per minute.
        :param approach: The approach mode, either 'linear', 'no overshoot' or
            'oscillate'.
        :param mode: The state of the magnet at the end of the charging
            process, either 'persistent' or 'driven'.
        :param wait_for_stability: If `True`, the function call blocks until
            the target field is reached and stable.
        :param delay: Specifies the frequency in seconds how often the magnet
            status is checked. (This has no effect if wait_for_stability is
            `False`).

        """
        self.target_field = field, rate, approach, mode
        if wait_for_stability and self.system_status['magnet'].startswith('persist'):
            # Wait until the persistent switch heats up.
            time.sleep(self.magnet_config[5])

        while wait_for_stability:
            status = self.system_status['magnet']
            if status in ('persistent, stable', 'driven, stable'):
                break
            time.sleep(delay)