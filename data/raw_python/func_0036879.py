def load_vcoef(filename):
    """Loads a set of vector coefficients that were saved in MATLAB. The third
    number on the first line is the directivity calculated within the MATLAB
    code."""

    with open(filename) as f: 
        lines = f.readlines()

        lst = lines[0].split(',')

        nmax = int(lst[0])
        mmax = int(lst[1])
        directivity = float(lst[2])

        L = (nmax + 1) + mmax * (2 * nmax - mmax + 1);

        vec1 = np.zeros(L, dtype=np.complex128)
        vec2 = np.zeros(L, dtype=np.complex128)
   
        lines.pop(0)

        n = 0
        in_vec2 = False
        for line in lines:

            if line.strip() == 'break':
                n = 0
                in_vec2 = True;
            else:
                lst = line.split(',')
                re = float(lst[0])
                im = float(lst[1])

                if in_vec2:
                    vec2[n] = re + 1j * im
                else:
                    vec1[n] = re + 1j * im

                n += 1

    return (sp.VectorCoefs(vec1, vec2, nmax, mmax), directivity)