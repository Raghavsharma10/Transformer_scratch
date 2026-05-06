def define(cls, expr):
        """
        GENERAL SUPPORT FOR BUILDING EXPRESSIONS FROM JSON EXPRESSIONS
        OVERRIDE THIS IF AN OPERATOR EXPECTS COMPLICATED PARAMETERS
        :param expr: Data representing a JSON Expression
        :return: parse tree
        """

        try:
            lang = cls.lang
            items = items_(expr)
            for item in items:
                op, term = item
                full_op = operators.get(op)
                if full_op:
                    class_ = lang.ops[full_op.id]
                    clauses = {k: jx_expression(v) for k, v in expr.items() if k != op}
                    break
            else:
                if not items:
                    return NULL
                raise Log.error("{{operator|quote}} is not a known operator", operator=expr)

            if term == None:
                return class_([], **clauses)
            elif is_list(term):
                terms = [jx_expression(t) for t in term]
                return class_(terms, **clauses)
            elif is_data(term):
                items = items_(term)
                if class_.has_simple_form:
                    if len(items) == 1:
                        k, v = items[0]
                        return class_([Variable(k), Literal(v)], **clauses)
                    else:
                        return class_({k:  Literal(v) for k, v in items}, **clauses)
                else:
                    return class_(_jx_expression(term, lang), **clauses)
            else:
                if op in ["literal", "date", "offset"]:
                    return class_(term, **clauses)
                else:
                    return class_(_jx_expression(term, lang), **clauses)
        except Exception as e:
            Log.error("programmer error expr = {{value|quote}}", value=expr, cause=e)