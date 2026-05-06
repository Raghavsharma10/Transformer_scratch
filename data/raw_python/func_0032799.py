def currentPage(self):
        """
        Return a sequence of mappings of attribute IDs to column values, to
        display to the user.

        nextPage/prevPage will strive never to skip items whose column values
        have not been returned by this method.

        This is best explained by a demonstration.  Let's say you have a table
        viewing an item with attributes 'a' and 'b', like this:

        oid | a | b
        ----+---+--
        0   | 1 | 2
        1   | 3 | 4
        2   | 5 | 6
        3   | 7 | 8
        4   | 9 | 0

        The table has 2 items per page.  You call currentPage and receive a
        page which contains items oid 0 and oid 1.  item oid 1 is deleted.

        If the next thing you do is to call nextPage, the result of currentPage
        following that will be items beginning with item oid 2.  This is
        because although there are no longer enough items to populate a full
        page from 0-1, the user has never seen item #2 on a page, so the 'next'
        page from the user's point of view contains #2.

        If instead, at that same point, the next thing you did was to call
        currentPage, *then* nextPage and currentPage again, the first
        currentPage results would contain items #0 and #2; the following
        currentPage results would contain items #3 and #4.  In this case, the
        user *has* seen #2 already, so the user expects to see the following
        item, not the same item again.
        """

        self._updateResults(self._sortAttributeValue(0), equalToStart=True, refresh=True)
        return self._currentResults