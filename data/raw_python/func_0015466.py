def _gen_schema_xref(self, expr: Optional[Union[ShExJ.shapeExprLabel, ShExJ.shapeExpr]]) -> None:
        """
        Generate the schema_id_map

        :param expr: root shape expression
        """
        if expr is not None and not isinstance_(expr, ShExJ.shapeExprLabel) and 'id' in expr and expr.id is not None:
            abs_id = self._resolve_relative_uri(expr.id)
            if abs_id not in self.schema_id_map:
                self.schema_id_map[abs_id] = expr
                if isinstance(expr, (ShExJ.ShapeOr, ShExJ.ShapeAnd)):
                    for expr2 in expr.shapeExprs:
                        self._gen_schema_xref(expr2)
                elif isinstance(expr, ShExJ.ShapeNot):
                    self._gen_schema_xref(expr.shapeExpr)
                elif isinstance(expr, ShExJ.Shape):
                    if expr.expression is not None:
                        self._gen_te_xref(expr.expression)