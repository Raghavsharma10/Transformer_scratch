def discard(self, element):
        """Remove element from the RangeSet if it is a member.

        If the element is not a member, do nothing.
        """
        try:
            i = int(element)
            set.discard(self, i)
        except ValueError:
            pass