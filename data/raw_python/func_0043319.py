def get_expr(self, ctx):
        """
        Returns the MUF needed to get the contents of the lvalue.
        Returned MUF will push the contained value onto the stack.
        """
        varname = ctx.lookup_variable(self.varname)
        if varname is None:
            val = ctx.lookup_constant(self.varname)
            if val:
                try:
                    return val.generate_code(ctx)
                except AttributeError:
                    return val
            raise MuvError(
                "Undeclared identifier '%s'." % self.varname,
                position=self.position
            )
        if len(self.indexing) == 0:
            return "{var} @".format(
                var=varname,
            )
        if len(self.indexing) == 1:
            return "{var} @ {idx} []".format(
                var=varname,
                idx=self.indexing[0],
            )
        return (
            "{var} @ {{ {idx} }}list array_nested_get".format(
                var=varname,
                idx=" ".join(str(x) for x in self.indexing),
            )
        )