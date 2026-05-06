def assert_type_and_length(varname, var, T, L = None, minL = None, maxL = None):
    'Facilitates simultaneous or one-line type/length checks.'
    if not isinstance(var, T):
        raise TypeError("Variable '{}' is supposed to be type '{}' but is '{}'".format(varname, T, type(var)))
    if isinstance(L, int):
        if not L == len(var):
            raise ValueError("Variable '{}' is supposed to be length {} but is {}".format(varname, L, len(var)))
    if isinstance(maxL, int):
        if maxL < len(var):
            raise ValueError("Variable '{}' is supposed to be smaller than {} but is length {}".format(varname, maxL, len(var)))
    if isinstance(minL, int):
        if minL > len(var):
            raise ValueError("Variable '{}' is supposed to be larger than {} but is length {}".format(varname, minL, len(var)))