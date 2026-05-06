def naturalize_thing(self, string):
        """
        Make a naturalized version of a general string, not a person's name.
        e.g., title of a book, a band's name, etc.

        string -- a lowercase string.
        """

        # Things we want to move to the back of the string:
        articles = [
                        'a', 'an', 'the',
                        'un', 'une', 'le', 'la', 'les', "l'", "l’",
                        'ein', 'eine', 'der', 'die', 'das',
                        'una', 'el', 'los', 'las',
                    ]

        sort_string = string
        parts = string.split(' ')

        if len(parts) > 1 and parts[0] in articles:
            if parts[0] != parts[1]:
                # Don't do this if the name is 'The The' or 'La La Land'.
                # Makes 'long blondes, the':
                sort_string = '{}, {}'.format(' '.join(parts[1:]), parts[0])

        sort_string = self._naturalize_numbers(sort_string)

        return sort_string