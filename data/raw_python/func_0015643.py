def insert(self, iter, text, length=-1):
        """insert(iter, text, length=-1)

        {{ all }}
        """

        Gtk.TextBuffer.insert(self, iter, text, length)