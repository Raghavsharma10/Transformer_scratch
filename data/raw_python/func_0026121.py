def watchTextSelection(self, event=None):
        """ Callback used to see if there is a new text selection. In certain
        cases we manually add the text to the clipboard (though on most
        platforms the correct behavior happens automatically). """
        # Note that this isn't perfect - it is a key click behind when
        # selections are made via shift-arrow.  If this becomes important, it
        # can likely be fixed with after().
        if self.entry.selection_present(): # entry must be text entry type
            i1 = self.entry.index(SEL_FIRST)
            i2 = self.entry.index(SEL_LAST)
            if i1 >= 0 and i2 >= 0 and i2 > i1:
                sel = self.entry.get()[i1:i2]
                # Add to clipboard on platforms where necessary.
                print('selected: "'+sel+'"')