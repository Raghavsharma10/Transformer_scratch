def plot(nxG, nyG, iBeg, iEnd, jBeg, jEnd, data, title=''):
    """
    Plot distributed array
    @param nxG number of global cells in x
    @param nyG number of global cells in y
    @param iBeg global starting index in x
    @param iEnd global ending index in x
    @param jBeg global starting index in y
    @param jEnd global ending index in y
    @param data local array
    @param title plot title
    """
    sz = MPI.COMM_WORLD.Get_size()
    rk = MPI.COMM_WORLD.Get_rank()
    iBegs = MPI.COMM_WORLD.gather(iBeg, root=0)
    iEnds = MPI.COMM_WORLD.gather(iEnd, root=0)
    jBegs = MPI.COMM_WORLD.gather(jBeg, root=0)
    jEnds = MPI.COMM_WORLD.gather(jEnd, root=0)
    arrays = MPI.COMM_WORLD.gather(numpy.array(data), root=0)
    if rk == 0:
        bigArray = numpy.zeros((nxG, nyG), data.dtype)
        for pe in range(sz):
            bigArray[iBegs[pe]:iEnds[pe], jBegs[pe]:jEnds[pe]] = arrays[pe]
        from matplotlib import pylab
        pylab.pcolor(bigArray.transpose())
        # add the decomp domains
        for pe in range(sz):
            pylab.plot([iBegs[pe], iBegs[pe]], [0, nyG - 1], 'w--')
            pylab.plot([0, nxG - 1], [jBegs[pe], jBegs[pe]], 'w--')
        pylab.title(title)
        pylab.show()