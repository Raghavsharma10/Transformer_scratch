def _genKeysBins(self):
        """ Generates keys from bins, sets self._allowedKeys normally set in _classVariables
        """
        binlimits = self._binlimits

        allowedKeys = []
        midbinlimits = binlimits

        if binlimits[0] == -float('inf'):
            midbinlimits = binlimits[1:]  # remove the bottom limit
            allowedKeys.append('<{0}'.format(midbinlimits[0]))

        if binlimits[-1] == float('inf'):
            midbinlimits = midbinlimits[:-1]

        lastbin = midbinlimits[0]

        for binlimit in midbinlimits[1:]:
            if lastbin == binlimit:
                allowedKeys.append('{0}'.format(binlimit))
            else:
                allowedKeys.append('{0} to {1}'.format(lastbin, binlimit))
            lastbin = binlimit

        if binlimits[-1] == float('inf'):
            allowedKeys.append('{0}+'.format(binlimits[-2]))

        allowedKeys.append('Uncertain')
        self._allowedKeys = allowedKeys