def RSA(m1,m2):
    '''RSA analysis will compare the similarity of two matrices
    '''
    from scipy.stats import pearsonr
    import scipy.linalg
    import numpy

    # This will take the diagonal of each matrix (and the other half is changed to nan) and flatten to vector
    vectorm1 = m1.mask(numpy.triu(numpy.ones(m1.shape)).astype(numpy.bool)).values.flatten()
    vectorm2 = m2.mask(numpy.triu(numpy.ones(m2.shape)).astype(numpy.bool)).values.flatten()
    # Now remove the nans
    m1defined = numpy.argwhere(~numpy.isnan(numpy.array(vectorm1,dtype=float)))
    m2defined = numpy.argwhere(~numpy.isnan(numpy.array(vectorm2,dtype=float)))
    idx = numpy.intersect1d(m1defined,m2defined)
    return pearsonr(vectorm1[idx],vectorm2[idx])[0]