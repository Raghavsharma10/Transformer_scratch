def unary_set_expr(self, ctx, oper, postoper=False):
        """
        Returns the MUF needed to do an unary operation on the lvalue. (++, --.)
        """
        if self.readonly:
            raise MuvError(
                "Cannot increment or decrement constant '%s'." % self.varname,
                position=self.position
            )
        varname = ctx.lookup_variable(self.varname)
        if varname is None:
            raise MuvError(
                "Undeclared identifier '%s'." % self.varname,
                position=self.position
            )
        if len(self.indexing) == 0:
            if postoper:
                fmt = "{var} @ {var} {oper}"
            else:
                fmt = "{var} dup {oper} @"
            return fmt.format(var=varname, oper=oper)
        if len(self.indexing) == 1:
            if ctx.target in ['fb7']:
                if postoper:
                    fmt = (
                        "{idx} {var} @ "
                        "dup 3 pick [] "
                        "dup -4 rotate {oper} "
                        "swap rot ->[] pop"
                    )
                else:
                    fmt = (
                        "{idx} {var} @ "
                        "dup 3 pick [] {oper} "
                        "dup -4 rotate "
                        "swap rot ->[] pop"
                    )
            else:
                if postoper:
                    fmt = (
                        "{idx} {var} @ "
                        "dup 3 pick [] "
                        "dup -4 rotate {oper} "
                        "swap rot ->[] {var} !"
                    )
                else:
                    fmt = (
                        "{idx} {var} @ "
                        "dup 3 pick [] {oper} "
                        "dup -4 rotate "
                        "swap rot ->[] {var} !"
                    )
            return fmt.format(
                var=varname,
                oper=oper,
                idx=self.indexing[0].generate_code(ctx),
            )
        if ctx.target in ['fb7']:
            if postoper:
                fmt = (
                    "{{ {idx} }}list {var} @ "
                    "dup 3 pick array_nested_get "
                    "dup -4 rotate {oper} "
                    "swap rot array_nested_set pop"
                )
            else:
                fmt = (
                    "{{ {idx} }}list {var} @ "
                    "dup 3 pick array_nested_get {oper} "
                    "dup -4 rotate "
                    "swap rot array_nested_set pop"
                )
        else:
            if postoper:
                fmt = (
                    "{{ {idx} }}list {var} @ "
                    "dup 3 pick array_nested_get "
                    "dup -4 rotate {oper} "
                    "swap rot array_nested_set {var} !"
                )
            else:
                fmt = (
                    "{{ {idx} }}list {var} @ "
                    "dup 3 pick array_nested_get {oper} "
                    "dup -4 rotate "
                    "swap rot array_nested_set {var} !"
                )
        return fmt.format(
            var=varname,
            oper=oper,
            idx=" ".join(
                x.generate_code(ctx)
                for x in self.indexing
            )
        )