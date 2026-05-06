def _eval(self):
        "Evaluates a individual using recursion and self._pos as pointer"
        pos = self._pos
        self._pos += 1
        node = self._ind[pos]
        if isinstance(node, Function):
            args = [self._eval() for x in range(node.nargs)]
            node.eval(args)
            for x in args:
                x.hy = None
                x.hy_test = None
        else:
            node.eval(self._X)
        return node