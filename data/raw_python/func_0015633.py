def insert_text(self, text, position):
        """insert_text(self, text, position)

        :param new_text:
            the text to append
        :type new_text: :obj:`str`

        :param position:
            location of the position text will be inserted at
        :type position: :obj:`int`

        :returns:
            location of the position text will be inserted at
        :rtype: :obj:`int`

        Inserts `new_text` into the contents of the
        widget, at position `position`.

        Note that the position is in characters, not in bytes.
        """

        return super(Editable, self).insert_text(text, -1, position)