def sweep(self, mode, speed=None):
        """Starts the output current sweep.

        :param mode: The sweep mode. Valid entries are `'UP'`, `'DOWN'`,
            `'PAUSE'`or `'ZERO'`. If in shim mode, `'LIMIT'` is valid as well.
        :param speed: The sweeping speed. Valid entries are `'FAST'`, `'SLOW'`
            or `None`.

        """
        sweep_modes = ['UP', 'DOWN', 'PAUSE', 'ZERO', 'LIMIT']
        sweep_speed = ['SLOW', 'FAST', None]
        if not mode in sweep_modes:
            raise ValueError('Invalid sweep mode.')
        if not speed in sweep_speed:
            raise ValueError('Invalid sweep speed.')
        if speed is None:
            self._write('SWEEP {0}'.format(mode))
        else:
            self._write('SWEEP {0} {1}'.format(mode, speed))