def nEx(mt, x, n):
    """ nEx : Returns the EPV of a pure endowment (deferred capital). 
    Pure endowment benefits are conditional on the survival of the policyholder. (v^n * npx) """
    return mt.Dx[x + n] / mt.Dx[x]