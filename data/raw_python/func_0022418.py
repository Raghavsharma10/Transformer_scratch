def fling_forward_horizontally(self, *args, **selectors):
        """
        Perform fling forward (horizontally)action on the object which has *selectors* attributes.

        Return whether the object can be fling or not.
        """
        return self.device(**selectors).fling.horiz.forward()