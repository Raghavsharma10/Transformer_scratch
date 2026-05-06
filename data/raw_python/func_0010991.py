def _eval(self, node):
        """
        Evaluate a node

        :param node: Node to eval
        :return: Result of node
        """
        try:
            handler = self.nodes[type(node)]
        except KeyError:
            raise ValueError("Sorry, {0} is not available in this evaluator".format(type(node).__name__))

        return handler(node)