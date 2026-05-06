def round_sig_error(x, ex, n, paren=False):
    '''Find ex rounded to n sig-figs and make the floating point x
    match the number of decimals.  If [paren], the string is
    returned as quantity(error) format'''
    stex = round_sig(ex,n)
    if stex.find('.') < 0:
        extra_zeros = len(stex) - n
        sigfigs = len(str(int(x))) - extra_zeros
        stx = round_sig(x,sigfigs)
    else:
        num_after_dec = len(string.split(stex,'.')[1])
        stx = ("%%.%df" % num_after_dec) % (x)
    if paren:
        if stex.find('.') >= 0:
            stex = stex[stex.find('.')+1:]
        return "%s(%s)" % (stx,stex)
    return stx,stex