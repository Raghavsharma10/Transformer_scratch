def focusPrev(self, event):
        """Set focus to previous item in sequence"""
        try:
            event.widget.tk_focusPrev().focus_set()
        except TypeError:
            # see tkinter equivalent code for tk_focusPrev to see
            # commented original version
            name = event.widget.tk.call('tk_focusPrev', event.widget._w)
            event.widget._nametowidget(str(name)).focus_set()