def rebin(a, newshape):
    """
    Auxiliary function to rebin ndarray data.
    Source : http://www.scipy.org/Cookbook/Rebinning
        example usage:
        >>> a=rand(6,4); b=rebin(a,(3,2))
        """
        
    shape = a.shape

    lenShape = len(shape)

    factor = np.asarray(shape)/np.asarray(newshape)
    #print factor

    evList = ['a.reshape('] + \
             ['newshape[%d],factor[%d],'%(i,i) for i in xrange(lenShape)] + \
             [')'] + ['.sum(%d)'%(i+1) for i in xrange(lenShape)] + \
             ['/factor[%d]'%i for i in xrange(lenShape)]

    return eval(''.join(evList))