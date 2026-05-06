def _visit_te_shape(self, shape: ShExJ.shapeExpr, visit_center: _VisitorCenter) -> None:
        """
        Visit a shape expression that was reached through a triple expression.  This, in turn, is used to visit
        additional triple expressions that are referenced by the Shape

        :param shape: Shape reached through triple expression traverse
        :param visit_center: context used in shape visitor
        """
        if isinstance(shape, ShExJ.Shape) and shape.expression is not None:
            visit_center.f(visit_center.arg_cntxt, shape.expression, self)