def scroll_forward_horizontally(self, steps=10, *args, **selectors):
        """
        Perform scroll forward (horizontally)action on the object which has *selectors* attributes.

        Return whether the object can be Scroll or not.

        See `Scroll Forward Vertically` for more details.
        """
        return self.device(**selectors).scroll.horiz.forward(steps=steps)