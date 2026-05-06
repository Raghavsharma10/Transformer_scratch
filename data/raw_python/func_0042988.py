def convert(self, expr):
        """
        EXPAND INSTANCES OF name TO value
        """
        if expr is True or expr == None or expr is False:
            return expr
        elif is_number(expr):
            return expr
        elif expr == ".":
            return "."
        elif is_variable_name(expr):
            return coalesce(self.dimensions[expr], expr)
        elif is_text(expr):
            Log.error("{{name|quote}} is not a valid variable name", name=expr)
        elif isinstance(expr, Date):
            return expr
        elif is_op(expr, QueryOp):
            return self._convert_query(expr)
        elif is_data(expr):
            if expr["from"]:
                return self._convert_query(expr)
            elif len(expr) >= 2:
                #ASSUME WE HAVE A NAMED STRUCTURE, NOT AN EXPRESSION
                return wrap({name: self.convert(value) for name, value in expr.leaves()})
            else:
                # ASSUME SINGLE-CLAUSE EXPRESSION
                k, v = expr.items()[0]
                return converter_map.get(k, self._convert_bop)(self, k, v)
        elif is_many(expr):
            return wrap([self.convert(value) for value in expr])
        else:
            return expr