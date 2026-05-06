def merge_pres_feats(pres, features):
    """
    Helper function to merge pres and features to support legacy features argument
    """

    sub = []
    for psub, fsub in zip(pres, features):
        exp = []
        for pexp, fexp in zip(psub, fsub):
            lst = []
            for p, f in zip(pexp, fexp):
                p.update(f)
                lst.append(p)
            exp.append(lst)
        sub.append(exp)
    return sub