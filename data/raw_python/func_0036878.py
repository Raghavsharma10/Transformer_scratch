def load_coef(filename):
    """Loads a file that was saved with save_coef."""
    
    with open(filename) as f: 
        lines = f.readlines()

        lst = lines[0].split(',')

        nmax = int(lst[0])
        mmax = int(lst[1])

        L = (nmax + 1) + mmax * (2 * nmax - mmax + 1);

        vec = np.zeros(L, dtype=np.complex128)
   
        lines.pop(0)

        for n, line in enumerate(lines):
            lst = line.split(',')
            re = float(lst[0])
            im = float(lst[1])
            vec[n] = re + 1j * im

    return sp.ScalarCoefs(vec, nmax, mmax)