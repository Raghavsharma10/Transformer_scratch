def swipe_right(self, steps=10, *args, **selectors):
        """
        Swipe the UI object with *selectors* from center to right

        See `Swipe Left` for more details.
        """
        self.device(**selectors).swipe.right(steps=steps)