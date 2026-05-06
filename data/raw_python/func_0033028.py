def snap(self, *args):
        """Records up to 6 parameters at a time.

        :param args: Specifies the values to record. Valid ones are 'X', 'Y',
          'R', 'theta', 'AuxIn1', 'AuxIn2', 'AuxIn3', 'AuxIn4', 'Ref', 'CH1'
          and 'CH2'. If none are given 'X' and 'Y' are used.

        """
        # TODO: Do not use transport directly.
        params = {'X': 1, 'Y': 2, 'R': 3, 'Theta': 4, 'AuxIn1': 5, 'AuxIn2': 6,
                  'AuxIn3': 7, 'AuxIn4': 8, 'Ref': 9, 'CH1': 10, 'CH2': 11}
        if not args:
            args = ['X', 'Y']
        if len(args) > 6:
            raise ValueError('Too many parameters (max: 6).')
        cmd = 'SNAP? ' + ','.join(map(lambda x: str(params[x]), args))
        result = self.transport.ask(cmd)
        return map(float, result.split(','))