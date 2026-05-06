def _jx_expression(expr, lang):
    """
    WRAP A JSON EXPRESSION WITH OBJECT REPRESENTATION
    """
    if is_expression(expr):
        # CONVERT TO lang
        new_op = lang[expr.id]
        if not new_op:
            # CAN NOT BE FOUND, TRY SOME PARTIAL EVAL
            return language[expr.id].partial_eval()
        return expr
        # return new_op(expr.args)  # THIS CAN BE DONE, BUT IT NEEDS MORE CODING, AND I WOULD EXPECT IT TO BE SLOW

    if expr is None:
        return TRUE
    elif expr in (True, False, None) or expr == None or isinstance(expr, (float, int, Decimal, Date)):
        return Literal(expr)
    elif is_text(expr):
        return Variable(expr)
    elif is_sequence(expr):
        return lang[TupleOp([_jx_expression(e, lang) for e in expr])]

    # expr = wrap(expr)
    try:
        items = items_(expr)

        for op, term in items:
            # ONE OF THESE IS THE OPERATOR
            full_op = operators.get(op)
            if full_op:
                class_ = lang.ops[full_op.id]
                if class_:
                    return class_.define(expr)

                # THIS LANGUAGE DOES NOT SUPPORT THIS OPERATOR, GOTO BASE LANGUAGE AND GET THE MACRO
                class_ = language[op.id]
                output = class_.define(expr).partial_eval()
                return _jx_expression(output, lang)
        else:
            if not items:
                return NULL
            raise Log.error("{{instruction|json}} is not known", instruction=items)

    except Exception as e:
        Log.error("programmer error expr = {{value|quote}}", value=expr, cause=e)