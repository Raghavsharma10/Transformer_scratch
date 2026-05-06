def visit_shapes(self, expr: ShExJ.shapeExpr, f: Callable[[Any, ShExJ.shapeExpr, "Context"], None], arg_cntxt: Any,
                     visit_center: _VisitorCenter = None, follow_inner_shapes: bool=True) -> None:
        """
        Visit expr and all of its "descendant" shapes.

        :param expr: root shape expression
        :param f: visitor function
        :param arg_cntxt: accompanying context for the visitor function
        :param visit_center: Recursive visit context.  (Not normally supplied on an external call)
        :param follow_inner_shapes: Follow nested shapes or just visit on outer level
        """
        if visit_center is None:
            visit_center = _VisitorCenter(f, arg_cntxt)
        has_id = getattr(expr, 'id', None) is not None
        if not has_id or not (visit_center.already_seen_shape(expr.id)
                              or visit_center.actively_visiting_shape(expr.id)):

            # Visit the root expression
            if has_id:
                visit_center.start_visiting_shape(expr.id)
            f(arg_cntxt, expr, self)

            # Traverse the expression and visit its components
            if isinstance(expr, (ShExJ.ShapeOr, ShExJ.ShapeAnd)):
                for expr2 in expr.shapeExprs:
                    self.visit_shapes(expr2, f, arg_cntxt, visit_center, follow_inner_shapes=follow_inner_shapes)
            elif isinstance(expr, ShExJ.ShapeNot):
                self.visit_shapes(expr.shapeExpr, f, arg_cntxt, visit_center, follow_inner_shapes=follow_inner_shapes)
            elif isinstance(expr, ShExJ.Shape):
                if expr.expression is not None and follow_inner_shapes:
                    self.visit_triple_expressions(expr.expression,
                                                  lambda ac, te, cntxt: self._visit_shape_te(te, visit_center),
                                                  arg_cntxt,
                                                  visit_center)
            elif isinstance_(expr, ShExJ.shapeExprLabel):
                if not visit_center.actively_visiting_shape(str(expr)) and follow_inner_shapes:
                    visit_center.start_visiting_shape(str(expr))
                    self.visit_shapes(self.shapeExprFor(expr), f, arg_cntxt, visit_center)
                    visit_center.done_visiting_shape(str(expr))
            if has_id:
                visit_center.done_visiting_shape(expr.id)