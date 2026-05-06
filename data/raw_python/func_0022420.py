def fling_forward_vertically(self, *args, **selectors):
        """
        Perform fling forward (vertically)action on the object which has *selectors* attributes.

        Return whether the object can be fling or not.
        """
        return self.device(**selectors).fling.vert.forward()