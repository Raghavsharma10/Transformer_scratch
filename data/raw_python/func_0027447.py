def attributes(self, *args, **kwargs):
        """
        Add one or more attributes to the :class:`xml4h.nodes.Element` node
        represented by this Builder.

        :return: the current Builder.

        Delegates to :meth:`xml4h.nodes.Element.set_attributes`.
        """
        self._element.set_attributes(*args, **kwargs)
        return self