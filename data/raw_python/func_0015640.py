def add_buttons(self, *args):
        """add_buttons(*args)

        The add_buttons() method adds several buttons to the Gtk.Dialog using
        the button data passed as arguments to the method. This method is the
        same as calling the Gtk.Dialog.add_button() repeatedly. The button data
        pairs - button text (or stock ID) and a response ID integer are passed
        individually. For example:

        .. code-block:: python

            dialog.add_buttons(Gtk.STOCK_OPEN, 42, "Close", Gtk.ResponseType.CLOSE)

        will add "Open" and "Close" buttons to dialog.
        """
        def _button(b):
            while b:
                t, r = b[0:2]
                b = b[2:]
                yield t, r

        try:
            for text, response in _button(args):
                self.add_button(text, response)
        except (IndexError):
            raise TypeError('Must pass an even number of arguments')