def to_log(x1, x2, x1err, x2err):
    """
    Take linear measurements and uncertainties and transform to log values.

    """
    logx1 = numpy.log10(numpy.array(x1))
    logx2 = numpy.log10(numpy.array(x2))
    x1err = numpy.log10(numpy.array(x1)+numpy.array(x1err)) - logx1
    x2err = numpy.log10(numpy.array(x2)+numpy.array(x2err)) - logx2
    return logx1, logx2, x1err, x2err