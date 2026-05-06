def _eval_call(self, node):
        """
        Evaluate a function call

        :param node: Node to eval
        :return: Result of node
        """
        try:
            func = self.functions[node.func.id]
        except KeyError:
            raise NameError(node.func.id)

        value = func(
            *(self._eval(a) for a in node.args),
            **dict(self._eval(k) for k in node.keywords)
        )

        if value is True:
            return 1
        elif value is False:
            return 0
        else:
            return value