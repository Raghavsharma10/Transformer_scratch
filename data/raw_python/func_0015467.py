def _gen_te_xref(self, expr: Union[ShExJ.tripleExpr, ShExJ.tripleExprLabel]) -> None:
        """
        Generate the triple expression map (te_id_map)

        :param expr: root triple expression

        """
        if expr is not None and not isinstance_(expr, ShExJ.tripleExprLabel) and 'id' in expr and expr.id is not None:
            if expr.id in self.te_id_map:
                return
            else:
                self.te_id_map[self._resolve_relative_uri(expr.id)] = expr
        if isinstance(expr, (ShExJ.OneOf, ShExJ.EachOf)):
            for expr2 in expr.expressions:
                self._gen_te_xref(expr2)
        elif isinstance(expr, ShExJ.TripleConstraint):
            if expr.valueExpr is not None:
                self._gen_schema_xref(expr.valueExpr)