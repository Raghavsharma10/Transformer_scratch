def _eval_unaryop(self, node):
        """
        Evaluate a unary operator node (ie. -2, +3)
        Currently just supports positive and negative

        :param node: Node to eval
        :return: Result of node
        """
        return self.operators[type(node.op)](self._eval(node.operand))