def naturalize_person(self, string):
        """
        Attempt to make a version of the string that has the surname, if any,
        at the start.

        'John, Brown' to 'Brown, John'
        'Sir John Brown Jr' to 'Brown, Sir John Jr'
        'Prince' to 'Prince'

        string -- The string to change.
        """
        suffixes = [
                    'Jr', 'Jr.', 'Sr', 'Sr.',
                    'I', 'II', 'III', 'IV', 'V',
                    ]
        # Add lowercase versions:
        suffixes = suffixes + [s.lower() for s in suffixes]

        # If a name has a capitalised particle in we use that to sort.
        # So 'Le Carre, John' but 'Carre, John le'.
        particles = [
                    'Le', 'La',
                    'Von', 'Van',
                    'Du', 'De',
                    ]

        surname = '' # Smith
        names = ''   # Fred James
        suffix = ''  # Jr

        sort_string = string
        parts = string.split(' ')

        if parts[-1] in suffixes:
            # Remove suffixes entirely, as we'll add them back on the end.
            suffix = parts[-1]
            parts = parts[0:-1] # Remove suffix from parts
            sort_string = ' '.join(parts)

        if len(parts) > 1:

            if parts[-2] in particles:
                # From ['Alan', 'Barry', 'Le', 'Carré']
                # to   ['Alan', 'Barry', 'Le Carré']:
                parts = parts[0:-2] + [ ' '.join(parts[-2:]) ]

            # From 'David Foster Wallace' to 'Wallace, David Foster':
            sort_string = '{}, {}'.format(parts[-1], ' '.join(parts[:-1]))

        if suffix:
            # Add it back on.
            sort_string = '{} {}'.format(sort_string, suffix)

        # In case this name has any numbers in it.
        sort_string = self._naturalize_numbers(sort_string)

        return sort_string