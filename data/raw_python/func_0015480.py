def satisfiesExternal(cntxt: Context, n: Node, se: ShExJ.ShapeExternal, c: DebugContext) -> bool:
    """ Se is a ShapeExternal and implementation-specific mechansims not defined in this specification indicate
     success.
     """
    if c.debug:
        print(f"id: {se.id}")
    extern_shape = cntxt.external_shape_for(se.id)
    if extern_shape:
        return satisfies(cntxt, n, extern_shape)
    cntxt.fail_reason = f"{se.id}: Shape is not in Schema"
    return False