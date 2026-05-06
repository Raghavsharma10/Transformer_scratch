def document(self):
        """
        :return: the :class:`Document` node that contains this node,
            or ``self`` if this node is the document.
        """
        if self.is_document:
            return self
        return self.adapter.wrap_document(self.adapter.impl_document)