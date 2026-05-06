def lookup(self, c, lookup_order="CF"):
        """
        Function to lookup a character in the casefold map.

        The casefold map has four sub-tables, the 'C' or common table, the 'F' or
        full table, the 'S' or simple table and the 'T' or the Turkic special
        case table.  These tables correspond to the statuses defined in the
        CaseFolding.txt file.  We can specify the order of the tables to use for
        performing the lookup by the lookup_order parameter.

        Per the usage specified in the CaseFolding.txt file, we can use the 'C'
        and 'S' tables for doing a simple case fold.  To perform a full case
        fold, we can use the 'C' and 'F' tables.  The default behavior for this
        function is a full case fold (lookup_order="CF").

        :param c: character to lookup
        :param lookup_order:
        """
        if not isinstance(c, six.text_type):
            raise TypeError(u"Character to lookup must be of type 'unicode'!")
        for d in lookup_order:
            try:
                return self._casefold_map[d][c]
            except KeyError:
                pass
        return c