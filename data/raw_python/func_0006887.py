def follow(self, chars):
        """
        Traverse the GADDAG to the node at the end of the given characters.

        Args:
            chars: An string of characters to traverse in the GADDAG.

        Returns:
            The Node which is found by traversing the tree.
        """
        chars = chars.lower()

        node = self.node
        for char in chars:
            node = cgaddag.gdg_follow_edge(self.gdg, node, char.encode("ascii"))
            if not node:
                raise KeyError(char)

        return Node(self.gdg, node)