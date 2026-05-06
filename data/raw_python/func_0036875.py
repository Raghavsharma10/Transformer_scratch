def save_coef(scoef, filename):
    """Saves ScalarCoeffs object 'scoef' to file. The first line of the 
    file has the max number N and the max number M of the scoef structure 
    separated by a comma. The remaining lines have the form
    
        3.14, 2.718

    The first number is the real part of the mode and the second is the 
    imaginary. 
    """
    
    nmax = scoef.nmax
    mmax = scoef.mmax

    frmstr = "{0:.16e},{1:.16e}\n"

    L = (nmax + 1) + mmax * (2 * nmax - mmax + 1);

    with open(filename, 'w') as f: 
        f.write("{0},{1}\n".format(nmax, mmax))
        for n in xrange(0, L):
            f.write(frmstr.format(scoef._vec[n].real,
                                  scoef._vec[n].imag))