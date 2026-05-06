def scroll_backward_vertically(self, steps=10, *args, **selectors):
        """
        Perform scroll backward (vertically)action on the object which has *selectors* attributes.

        Return whether the object can be Scroll or not.

        See `Scroll Forward Vertically` for more details.
        """
        return self.device(**selectors).scroll.vert.backward(steps=steps)