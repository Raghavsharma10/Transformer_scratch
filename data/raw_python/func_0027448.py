def processing_instruction(self, target, data):
        """
        Add a processing instruction node to the :class:`xml4h.nodes.Element`
        node represented by this Builder.

        :return: the current Builder.

        Delegates to :meth:`xml4h.nodes.Element.add_instruction`.
        """
        self._element.add_instruction(target, data)
        return self