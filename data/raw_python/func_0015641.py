def create_tag(self, tag_name=None, **properties):
        """Creates a tag and adds it to the tag table of the TextBuffer.

        :param str tag_name:
            Name of the new tag, or None
        :param **properties:
            Keyword list of properties and their values
        :returns:
            A new tag.

        This is equivalent to creating a Gtk.TextTag and then adding the
        tag to the buffer's tag table. The returned tag is owned by
        the buffer's tag table.

        If ``tag_name`` is None, the tag is anonymous.

        If ``tag_name`` is not None, a tag called ``tag_name`` must not already
        exist in the tag table for this buffer.

        Properties are passed as a keyword list of names and values (e.g.
        foreground='DodgerBlue', weight=Pango.Weight.BOLD)
        """

        tag = Gtk.TextTag(name=tag_name, **properties)
        self._get_or_create_tag_table().add(tag)
        return tag