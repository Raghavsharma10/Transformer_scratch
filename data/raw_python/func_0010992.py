def _eval_num(self, node):
        """
        Evaluate a numerical node

        :param node: Node to eval
        :return: Result of node
        """
        if self.floats:
            return node.n
        else:
            return int(node.n)