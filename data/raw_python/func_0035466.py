def filter(self, node, condition):
        """
        This method accepts a node and the condition function; a
        generator will be returned to yield the nodes that got matched
        by the condition.
        """

        if not isinstance(node, Node):
            raise TypeError('not a node')

        for child in node:
            if condition(child):
                yield child
            for subchild in self.filter(child, condition):
                yield subchild