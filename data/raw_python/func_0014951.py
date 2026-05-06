def reduceR(f, sequence, initial=__unique):
    """*R*ight reduce.
    >>> reduceR(lambda x,y:x/y, [1.,2.,3.,4]) == 1./(2./(3./4.)) == (1./2.)*(3./4.)
    True
    >>> reduceR(lambda x,y:x-y, iter([1,2,3]),4) == 1-(2-(3-4)) == (1-2)+(3-4)
    True
    """
    try: rev = reversed(sequence)
    except TypeError: rev = reversed(list(sequence))
    if initial is __unique: return reduce(lambda x,y:f(y,x), rev)
    else:                   return reduce(lambda x,y:f(y,x), rev, initial)