def _visit_shape_te(self, te: ShExJ.tripleExpr, visit_center: _VisitorCenter) -> None:
        """
        Visit a triple expression that was reached through a shape. This, in turn, is used to visit additional shapes
        that are referenced by a TripleConstraint
        :param te: Triple expression reached through a Shape.expression
        :param visit_center: context used in shape visitor
        """
        if isinstance(te, ShExJ.TripleConstraint) and te.valueExpr is not None:
            visit_center.f(visit_center.arg_cntxt, te.valueExpr, self)