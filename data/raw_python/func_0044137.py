def _guess_file_type(self, filename):
        """Attempt to guess the type of a MOC file.

        Returns "fits", "json" or "ascii" if successful and raised
        a ValueError otherwise.
        """

        # First attempt to guess from the file name.
        namelc = filename.lower()

        if namelc.endswith('.fits') or namelc.endswith('.fit'):
            return 'fits'
        elif namelc.endswith('.json'):
            return 'json'
        elif namelc.endswith('.txt') or namelc.endswith('.ascii'):
            return 'ascii'

        # Otherwise, if the file exists, look at the first character.
        if isfile(filename):
            with open(filename, 'r') as f:
                c = f.read(1)

            if c == 'S':
                return 'fits'
            elif c == '{':
                return 'json'
            elif c.isdigit():
                return 'ascii'

        raise ValueError('Unable to determine format of {0}'.format(filename))