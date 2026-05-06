def check(ty, val):
    "Checks that `val` adheres to type `ty`"
    
    if isinstance(ty, basestring):
        ty = Parser().parse(ty)

    return ty.enforce(val)