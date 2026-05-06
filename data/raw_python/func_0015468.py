def tripleExprFor(self, id_: ShExJ.tripleExprLabel) -> ShExJ.tripleExpr:
        """ Return the triple expression that corresponds to id """
        return self.te_id_map.get(id_)