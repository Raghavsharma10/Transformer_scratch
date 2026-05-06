def _handle_switch(self, node, scope, ctxt, stream):
        """Handle break node

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        def exec_case(idx, cases):
            # keep executing cases until a break is found,
            # or they've all been executed
            for case in cases[idx:]:
                stmts = case.stmts
                try:
                    for stmt in stmts:
                        self._handle_node(stmt, scope, ctxt, stream)
                except errors.InterpBreak as e:
                    break

        def get_stmts(stmts, res=None):
            if res is None:
                res = []

            stmts = self._flatten_list(stmts)
            for stmt in stmts:
                if isinstance(stmt, tuple):
                    stmt = stmt[1]

                res.append(stmt)

                if stmt.__class__ in [AST.Case, AST.Default]:
                    get_stmts(stmt.stmts, res)

            return res

        def get_cases(nodes, acc=None):
            cases = []

            stmts = get_stmts(nodes)
            for stmt in stmts:
                if stmt.__class__ in [AST.Case, AST.Default]:
                    cases.append(stmt)
                    stmt.stmts = []
                else:
                    cases[-1].stmts.append(stmt)

            return cases

        cond = self._handle_node(node.cond, scope, ctxt, stream)
    
        default_idx = None
        found_match = False

        cases = getattr(node, "pfp_cases", None)
        if cases is None:
            cases = get_cases(node.stmt.children())
            node.pfp_cases = cases

        for idx,child in enumerate(cases):
            if child.__class__ == AST.Default:
                default_idx = idx
                continue
            elif child.__class__ == AST.Case:
                expr = self._handle_node(child.expr, scope, ctxt, stream)
                if expr == cond:
                    found_match = True
                    exec_case(idx, cases)
                    break

        if default_idx is not None and not found_match:
            exec_case(default_idx, cases)