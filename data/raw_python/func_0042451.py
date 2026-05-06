def add_child(self, child):
        """
        Add a child to the tree. Extends discards all comments
        and whitespace Text. On non-whitespace Text, and any
        other nodes, raise a syntax error.
        """

        if isinstance(child, Comment):
            return

        # ignore Text nodes with whitespace-only content
        if isinstance(child, Text) and not child.text.strip():
            return

        super(Extends, self).add_child(child)