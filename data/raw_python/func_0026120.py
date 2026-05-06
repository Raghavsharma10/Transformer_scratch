def focusOut(self, event=None):
        """Clear selection (if text is selected in this widget)"""
        # do nothing if this isn't a text-enabled widget
        if not self.isSelectable:
            return
        if self.entryCheck(event) is None:
            # Entry value is OK
            # Save the last selection so it can be restored if we
            # come right back to this widget.  Then clear the selection
            # before moving on.
            entry = self.entry
            try:
                if not entry.selection_present():
                    self.lastSelection = None
                else:
                    self.lastSelection = (entry.index(SEL_FIRST),
                                          entry.index(SEL_LAST))
            except AttributeError:
                pass
            if USING_X and sys.platform == 'darwin':
                pass # do nothing here - we need it left selected for cut/paste
            else:
                entry.selection_clear()
        else:
            return "break"