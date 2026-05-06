def _eval_binop(self, node):
        """
        Evaluate a binary operator node (ie. 2+3, 5*6, 3 ** 4)

        :param node: Node to eval
        :return: Result of node
        """
        return self.operators[type(node.op)](self._eval(node.left),
                                             self._eval(node.right))