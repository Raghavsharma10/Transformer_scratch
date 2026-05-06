def element(self, *args, **kwargs):
        """
        Add a child element to the :class:`xml4h.nodes.Element` node
        represented by this Builder.

        :return: a new Builder that represents the child element.

        Delegates to :meth:`xml4h.nodes.Element.add_element`.
        """
        child_element = self._element.add_element(*args, **kwargs)
        return Builder(child_element)