def snap(self, *args):
        """Records multiple values at once.

        It takes two to six arguments specifying which values should be
        recorded together. Valid arguments are 'x', 'y', 'r', 'theta',
        'aux1', 'aux2', 'aux3', 'aux4', 'frequency', 'trace1', 'trace2',
        'trace3' and 'trace4'.

        snap is faster since it avoids communication overhead. 'x' and 'y'
        are recorded together, as well as 'r' and 'theta'. Between these
        pairs, there is a delay of approximately 10 us. 'aux1', 'aux2', 'aux3'
        and 'aux4' have am uncertainty of up to 32 us. It takes at least 40 ms
        or a period to calculate the frequency.

        E.g.::

            lockin.snap('x', 'theta', 'trace3')

        """
        length = len(args)
        if not 2 <= length <= 6:
            msg = 'snap takes 2 to 6 arguments, {0} given.'.format(length)
            raise TypeError(msg)
        # The program data type.
        param = Enum(
            'x', 'y', 'r', 'theta', 'aux1', 'aux2', 'aux3', 'aux4',
            'frequency', 'trace1', 'trace2', 'trace3', 'trace4'
        )
        # construct command,
        cmd = 'SNAP?', (Float,) * length, (param, ) * length
        return self._ask(cmd, *args)