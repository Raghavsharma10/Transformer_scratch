def _check_type(self, value):
        """Checks that *value* matches the type of this *Searcher*.

        Checks that *value* matches the type of this *Searcher*, returning the
        value if it does and raising a `TypeError` if it does not.

        :return: *value* if type of *value* matches type of this *Searcher*.
        :raises TypeError: if type of *value* does not match the type of this
            *Searcher*
        """
        if not isinstance(value, self.match_type):
            raise TypeError('Type ' + str(type(value)) + ' does not match '
                            'expected type ' + str(self.match_type))
        else:
            return value