def jx_expression_to_function(expr):
    """
    RETURN FUNCTION THAT REQUIRES PARAMETERS (row, rownum=None, rows=None):
    """
    if is_expression(expr):
        if is_op(expr, ScriptOp) and not is_text(expr.script):
            return expr.script
        else:
            return compile_expression(Python[expr].to_python())
    if (
        expr != None
        and not is_data(expr)
        and not is_list(expr)
        and hasattr(expr, "__call__")
    ):
        return expr
    return compile_expression(Python[jx_expression(expr)].to_python())