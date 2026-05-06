def load_chk(filename):
    '''Load a checkpoint file

       Argument:
        | filename  --  the file to load from

       The return value is a dictionary whose keys are field labels and the
       values can be None, string, integer, float, boolean or an array of
       strings, integers, booleans or floats.

       The file format is similar to the Gaussian fchk format, but has the extra
       feature that the shapes of the arrays are also stored.
    '''
    with open(filename) as f:
        result = {}
        while True:
            line = f.readline()
            if line == '':
                break
            if len(line) < 54:
                raise IOError('Header lines must be at least 54 characters long.')
            key = line[:40].strip()
            kind = line[47:52].strip()
            value = line[53:-1] # discard newline
            if kind == 'str':
                result[key] = value
            elif kind == 'int':
                result[key] = int(value)
            elif kind == 'bln':
                result[key] = value.lower() in ['true', '1', 'yes']
            elif kind == 'flt':
                result[key] = float(value)
            elif kind[3:5] == 'ar':
                if kind[:3] == 'str':
                    dtype = np.dtype('U22')
                elif kind[:3] == 'int':
                    dtype = int
                elif kind[:3] == 'bln':
                    dtype = bool
                elif kind[:3] == 'flt':
                    dtype = float
                else:
                    raise IOError('Unsupported kind: %s' % kind)
                shape = tuple(int(i) for i in value.split(','))
                array = np.zeros(shape, dtype)
                if array.size > 0:
                    work = array.ravel()
                    counter = 0
                    while True:
                        short = f.readline().split()
                        if len(short) == 0:
                            raise IOError('Insufficient data')
                        for s in short:
                            if dtype == bool:
                                work[counter] = s.lower() in ['true', '1', 'yes']
                            elif callable(dtype):
                                work[counter] = dtype(s)
                            else:
                                work[counter] = s
                            counter += 1
                            if counter == array.size:
                                break
                        if counter == array.size:
                            break
                result[key] = array
            elif kind == 'none':
                result[key] = None
            else:
                raise IOError('Unsupported kind: %s' % kind)
    return result