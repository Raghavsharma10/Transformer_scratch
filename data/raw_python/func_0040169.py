def _get_conditions(pk_conds, and_conds=None):
    """If and_conds = [a1, a2, ..., an] and pk_conds = [[b11, b12, ..., b1m], ... [bk1, ..., bkm]],
    this function will return the mysql condition clause:
        a1 & a2 & ... an & ((b11 and ... b1m) or ... (b11 and ... b1m))

    :param pk_conds: a list of list of primary key constraints returned by _get_conditions_list
    :param and_conds: additional and conditions to be placed on the query
    """
    if and_conds is None:
        and_conds = []

    if len(and_conds) == 0 and len(pk_conds) == 0:
        return sa.and_()

    condition1 = sa.and_(*and_conds)
    condition2 = sa.or_(*[sa.and_(*cond) for cond in pk_conds])
    return sa.and_(condition1, condition2)