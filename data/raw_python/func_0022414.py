def swipe_bottom(self, steps=10, *args, **selectors):
        """
        Swipe the UI object with *selectors* from center to bottom

        See `Swipe Left` for more details.
        """
        self.device(**selectors).swipe.down(steps=steps)