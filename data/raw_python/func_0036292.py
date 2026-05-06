def sbessely_array(xvec, N):
    """Outputs an array where each column is a vector of sbessel values. This
    is useful for plotting a set of Spherical Bessel Functions:

        A = sbessel.sbessely_array(np.linspace(.1,20,100),40)
        for sb in A:
            plot(sb)
        ylim((-.4,.4))
        show()
    """

    first_time = True  
    for x in xvec:
        a = sbessely(x, N)
        if first_time:
            out = np.array([a])
            first_time = False
        else:
            out = np.concatenate([out, [a]], axis=0)
            
    return out.T