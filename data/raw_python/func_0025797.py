def focusNext(self, event):
        """Set focus to next item in sequence"""
        try:
            event.widget.tk_focusNext().focus_set()
        except TypeError:
            # see tkinter equivalent code for tk_focusNext to see
            # commented original version
            name = event.widget.tk.call('tk_focusNext', event.widget._w)
            event.widget._nametowidget(str(name)).focus_set()