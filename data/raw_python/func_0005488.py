def round_sig_error2(x, ex1, ex2, n):
    '''Find min(ex1,ex2) rounded to n sig-figs and make the floating point x
    and max(ex,ex2) match the number of decimals.'''
    minerr = min(ex1,ex2)
    minstex = round_sig(minerr,n)
    if minstex.find('.') < 0:
        extra_zeros = len(minstex) - n
        sigfigs = len(str(int(x))) - extra_zeros
        stx = round_sig(x,sigfigs)
        maxstex = round_sig(max(ex1,ex2),sigfigs)
    else:
        num_after_dec = len(string.split(minstex,'.')[1])
        stx = ("%%.%df" % num_after_dec) % (x)
        maxstex = ("%%.%df" % num_after_dec) % (max(ex1,ex2))
    if ex1 < ex2:
        return stx,minstex,maxstex
    else:
        return stx,maxstex,minstex