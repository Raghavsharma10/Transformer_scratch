def rewrite_tg(env, tg_name, code):
    """Re-write a transform generating function pipe specification by extracting the tranform generating part,
    and replacing it with the generated transform. so:

       tgen(a,b,c).foo.bar

    becomes:

        tg = tgen(a,b,c)

        tg.foo.bar

    """

    visitor = ReplaceTG(env, tg_name)
    assert visitor.tg_name

    tree = visitor.visit(ast.parse(code))

    if visitor.loc:
        loc = ' #' + visitor.loc
    else:
        loc = file_loc()  # The AST visitor didn't match a call node

    if visitor.trans_gen:
        tg = meta.dump_python_source(visitor.trans_gen).strip()
    else:
        tg = None

    return meta.dump_python_source(tree).strip(), tg, loc