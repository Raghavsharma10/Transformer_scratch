def Else(self, *Else, **kwargs):
        """See `phi.dsl.Expression.If`"""
        root = self._root
        ast = self._ast

        next_else = E.Seq(*Else)._f
        ast = _add_else(ast, next_else)

        g = _compile_if(ast)

        return root.__then__(g, **kwargs)