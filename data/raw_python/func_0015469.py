def shapeExprFor(self, id_: Union[ShExJ.shapeExprLabel, START]) -> Optional[ShExJ.shapeExpr]:
        """ Return the shape expression that corresponds to id """
        rval = self.schema.start if id_ is START else self.schema_id_map.get(str(id_))
        return rval