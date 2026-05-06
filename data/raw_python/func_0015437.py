def matchesCardinality(cntxt: Context, T: RDFGraph, expr: Union[ShExJ.tripleExpr, ShExJ.tripleExprLabel],
                       c: DebugContext) -> bool:
    """ Evaluate cardinality expression

    expr has a cardinality of min and/or max not equal to 1, where a max of -1 is treated as unbounded, and
    T can be partitioned into k subsets T1, T2,…Tk such that min ≤ k ≤ max and for each Tn,
    matches(Tn, expr, m) by the remaining rules in this list.
    """
    # TODO: Cardinality defaults into spec
    min_ = expr.min if expr.min is not None else 1
    max_ = expr.max if expr.max is not None else 1

    cardinality_text = f"{{{min_},{'*' if max_ == -1 else max_}}}"
    if c.debug and (min_ != 0 or len(T) != 0):
        print(f"{cardinality_text} matching {len(T)} triples")
    if min_ == 0 and len(T) == 0:
        return True
    if isinstance(expr, ShExJ.TripleConstraint):
        if len(T) < min_:
            if len(T) > 0:
                _fail_triples(cntxt, T)
                cntxt.fail_reason = f"   {len(T)} triples less than {cardinality_text}"
            else:
                cntxt.fail_reason = f"   No matching triples found for predicate {cntxt.n3_mapper.n3(expr.predicate)}"
            return False
        elif 0 <= max_ < len(T):
            _fail_triples(cntxt, T)
            cntxt.fail_reason = f"   {len(T)} triples exceeds max {cardinality_text}"
            return False
        else:
            return all(matchesTripleConstraint(cntxt, t, expr) for t in T)
    else:
        for partition in _partitions(T, min_, max_):
            if all(matchesExpr(cntxt, part, expr) for part in partition):
                return True
        if min_ != 1 or max_ != 1:
            _fail_triples(cntxt, T)
            cntxt.fail_reason = f"   {len(T)} triples cannot be partitioned into {cardinality_text} passing groups"
        return False