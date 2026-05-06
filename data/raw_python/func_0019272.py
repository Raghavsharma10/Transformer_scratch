def wateryear(self):
        """The actual hydrological year according to the selected
        reference month.

        The reference mont reference |Date.refmonth| defaults to November:

        >>> october = Date('1996.10.01')
        >>> november = Date('1996.11.01')
        >>> october.wateryear
        1996
        >>> november.wateryear
        1997

        Note that changing |Date.refmonth| affects all |Date| objects:

        >>> october.refmonth = 10
        >>> october.wateryear
        1997
        >>> november.wateryear
        1997
        >>> october.refmonth = 'November'
        >>> october.wateryear
        1996
        >>> november.wateryear
        1997
        """
        if self.month < self._firstmonth_wateryear:
            return self.year
        return self.year + 1