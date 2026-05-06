def add(self, element, pad=0):
        """Add an element to a RangeSet.
        This has no effect if the element is already present.
        """
        set.add(self, int(element))
        if pad > 0 and self.padding is None:
            self.padding = pad