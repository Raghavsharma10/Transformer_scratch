def doScroll(self, event):
        """Scroll the panel down to ensure widget with focus to be visible

        Tracks the last widget that doScroll was called for and ignores
        repeated calls.  That handles the case where the focus moves not
        between parameter entries but to someplace outside the hierarchy.
        In that case the scrolling is not expected.

        Returns false if the scroll is ignored, else true.
        """
        canvas = self.top.f.canvas
        widgetWithFocus = event.widget
        if widgetWithFocus is self.lastFocusWidget:
            return FALSE
        self.lastFocusWidget = widgetWithFocus
        if widgetWithFocus is None:
            return TRUE
        # determine distance of widget from top & bottom edges of canvas
        y1 = widgetWithFocus.winfo_rooty()
        y2 = y1 + widgetWithFocus.winfo_height()
        cy1 = canvas.winfo_rooty()
        cy2 = cy1 + canvas.winfo_height()
        yinc = self.yscrollincrement
        if y1<cy1:
            # this will continue to work when integer division goes away
            sdist = int((y1-cy1-yinc+1.)/yinc)
            canvas.yview_scroll(sdist, "units")
        elif cy2<y2:
            sdist = int((y2-cy2+yinc-1.)/yinc)
            canvas.yview_scroll(sdist, "units")
        return TRUE