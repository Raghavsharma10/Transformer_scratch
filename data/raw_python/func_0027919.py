def _tupleCompare(tuple1, ineq, tuple2,
                 eq=lambda a,b: (a==b),
                 ander=AND,
                 orer=OR):
    """
    Compare two 'in-database tuples'.  Useful when sorting by a compound key
    and slicing into the middle of that query.
    """

    orholder = []
    for limit in range(len(tuple1)):
        eqconstraint = [
            eq(elem1, elem2) for elem1, elem2 in zip(tuple1, tuple2)[:limit]]
        ineqconstraint = ineq(tuple1[limit], tuple2[limit])
        orholder.append(ander(*(eqconstraint + [ineqconstraint])))
    return orer(*orholder)