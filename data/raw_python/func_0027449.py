def ns_prefix(self, prefix, ns_uri):
        """
        Set the namespace prefix of the :class:`xml4h.nodes.Element` node
        represented by this Builder.

        :return: the current Builder.

        Delegates to :meth:`xml4h.nodes.Element.set_ns_prefix`.
        """
        self._element.set_ns_prefix(prefix, ns_uri)
        return self