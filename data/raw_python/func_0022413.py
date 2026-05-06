def swipe_top(self, steps=10, *args, **selectors):
        """
        Swipe the UI object with *selectors* from center to top

        See `Swipe Left` for more details.
        """
        self.device(**selectors).swipe.up(steps=steps)