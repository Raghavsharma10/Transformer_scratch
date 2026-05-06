def add_text(self, text):
        """
        Add a text node to this element.

        Adding text with this method is subtly different from assigning a new
        text value with :meth:`text` accessor, because it "appends" to rather
        than replacing this element's set of text nodes.

        :param text: text content to add to this element.
        :param type: string or anything that can be coerced by :func:`unicode`.
        """
        if not isinstance(text, basestring):
            text = unicode(text)
        self._add_text(self.impl_node, text)