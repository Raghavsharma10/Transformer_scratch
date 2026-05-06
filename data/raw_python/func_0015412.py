def nodeSatisfiesDataType(cntxt: Context, n: Node, nc: ShExJ.NodeConstraint, c: DebugContext) -> bool:
    """ `5.4.3 Datatype Constraints <http://shex.io/shex-semantics/#datatype>`_

    For a node n and constraint value v, nodeSatisfies(n, v) if n is an Literal with the datatype v and, if v is in
    the set of SPARQL operand data types[sparql11-query], an XML schema string with a value of the lexical form of
    n can be cast to the target type v per XPath Functions 3.1 section 19 Casting[xpath-functions]. Only datatypes
    supported by SPARQL MUST be tested but ShEx extensions MAY add support for other datatypes.
    """
    if nc.datatype is None:
        return True
    if c.debug:
        print(f" Datatype: {nc.datatype}")
    if not isinstance(n, Literal):
        cntxt.fail_reason = f"Datatype constraint ({nc.datatype}) " \
            f"does not match {type(n).__name__} {cntxt.n3_mapper.n3(n)}"
        cntxt.dump_bnode(n)
        return False
    actual_datatype = _datatype(n)
    if actual_datatype == str(nc.datatype) or \
        (is_sparql_operand_datatype(nc.datatype) and can_cast_to(n, nc.datatype)):
        return True
    cntxt.fail_reason = f"Datatype mismatch - expected: {nc.datatype} actual: {actual_datatype}"
    return False