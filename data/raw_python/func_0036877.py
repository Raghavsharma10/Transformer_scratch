def load_vpatt(filename1, filename2):
    """Loads a VectorPatternUniform pattern that is saved between two files.
    """

    with open(filename1) as f: 
        lines = f.readlines()

        lst = lines[0].split(',')

        patt1 = np.zeros([int(lst[0]), int(lst[1])],
                        dtype=np.complex128)

        lines.pop(0)

        for line in lines:
            lst = line.split(',')
            n = int(lst[0])
            m = int(lst[1])
            re = float(lst[2])
            im = float(lst[3])
            patt1[n, m] = re + 1j * im

    with open(filename2) as f: 
        lines2 = f.readlines()

        lst = lines2[0].split(',')

        patt2 = np.zeros([int(lst[0]), int(lst[1])],
                        dtype=np.complex128)

        lines2.pop(0)

        for line in lines2:
            lst = line.split(',')
            n = int(lst[0])
            m = int(lst[1])
            re = float(lst[2])
            im = float(lst[3])
            patt2[n, m] = re + 1j * im

    return sp.VectorPatternUniform(patt1, patt2)