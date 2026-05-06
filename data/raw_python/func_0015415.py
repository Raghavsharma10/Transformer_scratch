def nodeSatisfiesValues(cntxt: Context, n: Node, nc: ShExJ.NodeConstraint, _c: DebugContext) -> bool:
    """ `5.4.5 Values Constraint <http://shex.io/shex-semantics/#values>`_

     For a node n and constraint value v, nodeSatisfies(n, v) if n matches some valueSetValue vsv in v.
    """
    if nc.values is None:
        return True
    else:
        if any(_nodeSatisfiesValue(cntxt, n, vsv) for vsv in nc.values):
            return True
        else:
            cntxt.fail_reason = f"Node: {cntxt.n3_mapper.n3(n)} not in value set:\n\t " \
                f"{as_json(cntxt.type_last(nc), indent=None)[:60]}..."
            return False