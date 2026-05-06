def convert_relational(relational):
    """Convert all inequalities to >=0 form.
    """
    rel = relational.rel_op
    if rel in ['==', '>=', '>']:
        return relational.lhs-relational.rhs
    elif rel in ['<=', '<']:
        return relational.rhs-relational.lhs
    else:
        raise Exception("The relational operation ' + rel + ' is not "
                        "implemented!")