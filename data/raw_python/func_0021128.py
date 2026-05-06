def _naturalize_numbers(self, string):
        """
        Makes any integers into very zero-padded numbers.
        e.g. '1' becomes '00000001'.
        """

        def naturalize_int_match(match):
            return '%08d' % (int(match.group(0)),)

        string = re.sub(r'\d+', naturalize_int_match, string)

        return string