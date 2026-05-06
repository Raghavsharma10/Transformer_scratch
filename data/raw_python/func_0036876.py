def load_patt(filename):
    """Loads a file that was saved with the save_patt routine."""

    with open(filename) as f: 
        lines = f.readlines()

        lst = lines[0].split(',')

        patt = np.zeros([int(lst[0]), int(lst[1])],
                        dtype=np.complex128)

        lines.pop(0)

        for line in lines:
            lst = line.split(',')
            n = int(lst[0])
            m = int(lst[1])
            re = float(lst[2])
            im = float(lst[3])
            patt[n, m] = re + 1j * im

    return sp.ScalarPatternUniform(patt, doublesphere=False)