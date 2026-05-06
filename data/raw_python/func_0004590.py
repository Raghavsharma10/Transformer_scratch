def process(fn, bound_names):
    """
    process automatic context variable capturing.
    return the transformed function and its ast.
    """
    if isinstance(fn, _AutoContext):
        fn = fn.fn

    # noinspection PyArgumentList,PyArgumentList
    if isinstance(fn, _FnCodeStr):

        if bound_names:
            assign_code_str = '{syms} = map(state.ctx.get, {names})'.format(
                syms=', '.join(bound_names) + ',', names=repr(bound_names))
        else:
             assign_code_str = ''

        code = "def {0}({1}):\n{2}".format(
            fn.fn_name, ", ".join(fn.fn_args),
            textwrap.indent(assign_code_str + '\n' + fn.code, "    "))

        module_ast = ast.parse(code, fn.filename)

        bound_name_line_inc = int(bool(bound_names)) + 1

        module_ast: ast.Module = ast.increment_lineno(
            module_ast, fn.lineno - bound_name_line_inc)

        fn_ast = module_ast.body[0]

        if isinstance(fn_ast.body[-1], ast.Expr):
            # auto addition of tail expr return
            # in rbnf you don't need to write return if the last statement in the end is an expression.
            it: ast.Expr = fn_ast.body[-1]

            fn_ast.body[-1] = ast.Return(
                lineno=it.lineno, col_offset=it.col_offset, value=it.value)

        filename = fn.filename
        name = fn.fn_name

        code_object = compile(module_ast, filename, "exec")

        local = {}
        # TODO: using types.MethodType here to create the function object leads to various problems.
        # Actually I don't really known why util now.
        exec(code_object, fn.namespace, local)
        ret = local[name]

    else:
        if not bound_names:
            return fn, get_ast(fn)

        code = fn.__code__
        assigns = ast.parse("ctx = state.ctx\n" + "\n".join(
            "{0} = ctx.get({0!r})".format(name) for name in bound_names))

        module_ast = get_ast(fn)
        fn_ast: ast.FunctionDef = module_ast.body[0]
        fn_ast.body = assigns.body + fn_ast.body
        module_ast = ast.Module([fn_ast])

        filename = code.co_filename
        name = code.co_name
        __defaults__ = fn.__defaults__
        __closure__ = fn.__closure__
        __globals__ = fn.__globals__

        code_object = compile(module_ast, filename, "exec")

        code_object = next(
            each for each in code_object.co_consts
            if isinstance(each, types.CodeType) and each.co_name == name)

        # noinspection PyArgumentList,PyUnboundLocalVariable
        ret = types.FunctionType(code_object, __globals__, name, __defaults__,
                                 __closure__)
    return ret, module_ast