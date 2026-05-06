def focusIn(self, event=None):
        """Select all text (if applicable) on taking focus"""
        try:
            # doScroll returns false if the call was ignored because the
            # last call also came from this widget.  That avoids unwanted
            # scrolls and text selection when the focus moves in and out
            # of the window.
            if self.doScroll(event):
                self.entry.selection_range(0, END) # select all text in widget
            else:
                # restore selection to what it was on the last FocusOut
                if self.lastSelection:
                    self.entry.selection_range(*self.lastSelection)
        except AttributeError:
            pass