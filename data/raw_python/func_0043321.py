def oper_set_expr(self, ctx, oper, val):
        """
        Returns the MUF needed to do an oper-assign on the lvalue. (+=, etc.)
        """
        if self.readonly:
            raise MuvError(
                "Cannot assign value to constant '%s'." % self.varname,
                position=self.position
            )
        varname = ctx.lookup_variable(self.varname)
        if varname is None:
            raise MuvError(
                "Undeclared identifier '%s'." % self.varname,
                position=self.position
            )
        if len(self.indexing) == 0:
            if ctx.assign_level > 1:
                fmt = "{var} @ {val} {oper} dup {var} !"
            else:
                fmt = "{var} @ {val} {oper} {var} !"
            return fmt.format(
                var=varname,
                oper=oper,
                val=val.generate_code(ctx),
            )
        if len(self.indexing) == 1:
            if ctx.target in ['fb7']:
                if ctx.assign_level > 1:
                    fmt = (
                        "{var} @ {idx} "
                        "over over [] {val} {oper} "
                        "dup -4 rotate "
                        "-rot ->[] pop"
                    )
                else:
                    fmt = (
                        "{var} @ {idx} "
                        "over over [] {val} {oper} "
                        "-rot ->[] pop"
                    )
            else:
                if ctx.assign_level > 1:
                    fmt = (
                        "{var} @ {idx} "
                        "over over [] {val} {oper} "
                        "dup -4 rotate "
                        "rot rot ->[] {var} !"
                    )
                else:
                    fmt = (
                        "{var} @ {idx} "
                        "over over [] {val} {oper} "
                        "rot rot ->[] {var} !"
                    )
            return fmt.format(
                var=varname,
                oper=oper,
                val=val.generate_code(ctx),
                idx=self.indexing[0].generate_code(ctx),
            )
        if ctx.target in ['fb7']:
            if ctx.assign_level > 1:
                fmt = (
                    "{var} @ {{ {sidx} }}list array_nested_get "
                    "{lidx} over over [] {val} {oper} "
                    "dup -4 rotate "
                    "-rot ->[] pop"
                )
            else:
                fmt = (
                    "{var} @ {{ {sidx} }}list array_nested_get "
                    "{lidx} over over [] {val} {oper} "
                    "-rot ->[] pop"
                )
        else:
            if ctx.assign_level > 1:
                fmt = (
                    "{var} @ {{ {idx} }}list "
                    "over over array_nested_get {val} {oper} "
                    "dup -4 rotate "
                    "rot rot array_nested_set {var} !"
                )
            else:
                fmt = (
                    "{var} @ {{ {idx} }}list "
                    "over over array_nested_get {val} {oper} "
                    "rot rot array_nested_set {var} !"
                )
        return fmt.format(
            var=varname,
            oper=oper,
            val=val.generate_code(ctx),
            idx=" ".join(
                x.generate_code(ctx)
                for x in self.indexing
            ),
            sidx=" ".join(
                x.generate_code(ctx)
                for x in self.indexing[:-1]
            ),
            lidx=self.indexing[-1].generate_code(ctx)
        )