def swipe_left(self, steps=10, *args, **selectors):
        """
        Swipe the UI object with *selectors* from center to left.

        Example:

        | Swipe Left | description=Home screen 3 |                           | # swipe the UI object left              |
        | Swipe Left | 5                         | description=Home screen 3 | # swipe the UI object left with steps=5 |

        See `introduction` for details about Identified UI object.
        """
        self.device(**selectors).swipe.left(steps=steps)