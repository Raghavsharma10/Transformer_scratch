def xform(self, number, relation):
        """
        Strip trailing zeros from a number according to config and relation.

        :param number: a number
        :type number: list of int
        :param int relation: the relation of the display value to the actual
        """

        # pylint: disable=too-many-boolean-expressions
        if (self.CONFIG.strip) or \
           (self.CONFIG.strip_exact and relation == 0) or \
           (self.CONFIG.strip_whole and relation == 0 and \
            all(x == 0 for x in number)):
            return Strip._strip_trailing_zeros(number)
        return number