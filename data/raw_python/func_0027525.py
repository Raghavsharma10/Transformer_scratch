def find_first(self, name=None, ns_uri=None):
        """
        Find the first :class:`Element` node descendant of this node that
        matches any optional constraints, or None if there are no matching
        elements.

        Delegates to :meth:`find` with ``first_only=True``.
        """
        return self.find(name=name, ns_uri=ns_uri, first_only=True)