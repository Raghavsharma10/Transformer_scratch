def parse_aid(self, text, default_key):
        """Parse argument text for aid.

        May retrieve the aid from search result tables as necessary.  aresults
        determines which search results to use by default; True means aresults
        is the default.

        The last aid when no aid has been parsed yet is undefined.

        The accepted formats, in order:

        Last AID:                .
        Explicit AID:            aid:12345
        Explicit result number:  key:12
        Default result number:   12

        """

        if default_key not in self:
            raise ResultKeyError(default_key)

        if text == '.':
            return self.last_aid
        elif text.startswith('aid:'):
            return int(text[len('aid:'):])

        if ':' in text:
            match = self._key_pattern.search(text)
            if not match:
                raise InvalidSyntaxError(text)
            key = match.group(1)
            number = match.group(2)
        else:
            key = default_key
            number = text
        try:
            number = int(number)
        except ValueError:
            raise InvalidSyntaxError(number)

        try:
            return self[key].get_aid(number)
        except KeyError:
            raise ResultKeyError(key)
        except IndexError:
            raise ResultNumberError(key, number)