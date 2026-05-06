def __experimental_range(start, stop, var, cond, loc={}):
    '''Utility function made to reproduce range() with unit integer step
       but with the added possibility of specifying a condition
       on the looping variable  (e.g. var % 2  == 0)
    '''
    locals().update(loc)
    if start < stop:
        for __ in range(start, stop):
            locals()[var] = __
            if eval(cond, globals(), locals()):
                yield __
    else:
        for __ in range(start, stop, -1):
            locals()[var] = __
            if eval(cond, globals(), locals()):
                yield __