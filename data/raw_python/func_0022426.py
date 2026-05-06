def scroll_to_horizontally(self, obj, *args,**selectors):
        """
        Scroll(horizontally) on the object: obj to specific UI object which has *selectors* attributes appears.

        Return true if the UI object, else return false.

        See `Scroll To Vertically` for more details.
        """
        return obj.scroll.horiz.to(**selectors)