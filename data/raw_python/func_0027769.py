def findFirst(self, tableClass, comparison=None,
                  offset=None, sort=None, default=None):
        """
        Usage::

            s.findFirst(tableClass [, query arguments except 'limit'])

        Example::

            class YourItemType(Item):
                a = integer()
                b = text()
                c = integer()
            ...
            it = s.findFirst(YourItemType,
                             AND(YourItemType.a == 1,
                                 YourItemType.b == u'2'),
                                 sort=YourItemType.c.descending)

        Search for an item with columns in the database that match the passed
        comparison, offset and sort, returning the first match if one is found,
        or the passed default (None if none is passed) if one is not found.
        """

        limit = 1
        for item in self.query(tableClass, comparison, limit, offset, sort):
            return item
        return default