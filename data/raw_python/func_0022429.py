def scroll_forward_vertically(self, steps=10, *args, **selectors):
        """
        Perform scroll forward (vertically)action on the object which has *selectors* attributes.

        Return whether the object can be Scroll or not.

        Example:
        | ${can_be_scroll} | Scroll Forward Vertically | className=android.widget.ListView       |                                   | # Scroll forward the UI object with class name |
        | ${can_be_scroll} | Scroll Forward Vertically | 100                                     | className=android.widget.ListView | # Scroll with steps |
        """
        return self.device(**selectors).scroll.vert.forward(steps=steps)