def save_patt(patt, filename):
    """Saves ScalarPatternUniform object 'patt' to file. The first line of the 
    file has the number of rows and the number of columns in the pattern 
    separated by a comma (single sphere data). The remaining lines have the 
    form:
    
        0, 1, 3.14, 2.718

    The first two numbers are index for the row and index for the column 
    respectively. The last two numbers are the real and imaginary part of the 
    number associated with the row and column.
    """

    nrows = patt.nrows
    ncols = patt.ncols
    frmstr = "{0},{1},{2:.16e},{3:.16e}\n"

    ar = patt.array

    with open(filename, 'w') as f: 
        f.write("{0},{1}\n".format(nrows, ncols))
        for nr in xrange(0, nrows):
            for nc in xrange(0, ncols):
                f.write(frmstr.format(nr, nc, ar[nr, nc].real,
                                            ar[nr, nc].imag))