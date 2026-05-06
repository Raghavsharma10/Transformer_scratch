def find_doc(self, name=None, ns_uri=None, first_only=False):
        """
        Find :class:`Element` node descendants of the document containing
        this node, with optional constraints to limit the results.

        Delegates to :meth:`find` applied to this node's owning document.
        """
        return self.document.find(name=name, ns_uri=ns_uri,
            first_only=first_only)