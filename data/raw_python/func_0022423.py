def scroll_to_end_horizontally(self, steps=10, *args, **selectors):
        """
        Scroll the object which has *selectors* attributes to *end* horizontally.

        See `Scroll Forward Vertically` for more details.
        """
        return self.device(**selectors).scroll.horiz.toEnd(steps=steps)