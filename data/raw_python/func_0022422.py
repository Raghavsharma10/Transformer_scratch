def scroll_to_beginning_horizontally(self, steps=10, *args,**selectors):
        """
        Scroll the object which has *selectors* attributes to *beginning* horizontally.

        See `Scroll Forward Vertically` for more details.
        """
        return self.device(**selectors).scroll.horiz.toBeginning(steps=steps)