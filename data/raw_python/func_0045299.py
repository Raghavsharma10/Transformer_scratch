def reconcile(constraint):
    '''
    Returns an assignment of type variable names to
    types that makes this constraint satisfiable, or a Refutation
    '''
    
    if isinstance(constraint.subtype, NamedType):
        if isinstance(constraint.supertype, NamedType):
            if constraint.subtype.name == constraint.supertype.name:
                return {}
            else:
                return Refutation('Cannot reconcile different atomic types: %s' % constraint)
        elif isinstance(constraint.supertype, Variable):
            return {constraint.supertype.name: constraint.subtype}
        else:
            return Refutation('Cannot reconcile atomic type with non-atomic type: %s' % constraint)

    elif isinstance(constraint.supertype, NamedType):
        if isinstance(constraint.subtype, NamedType):
            if constraint.subtype.name == constraint.supertype.name:
                return {}
            else:
                return Refutation('Cannot reconcile different atomic types: %s' % constraint)
        elif isinstance(constraint.subtype, Variable):
            return {constraint.subtype.name: constraint.supertype}
        else:
            return Refutation('Cannot reconcile non-atomic type with atomic type: %s' % constraint)

    elif isinstance(constraint.supertype, Union):
        # Lots of stuff could happen here; unsure if there's research to bring to bear
        if constraint.subtype in constraint.supertype.types:
            return {}

    return Stumper(constraint)