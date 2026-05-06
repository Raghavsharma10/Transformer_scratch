def dump_chk(filename, data):
    '''Dump a checkpoint file

       Argument:
        | filename  --  the file to write to
        | data  -- a dictionary whose keys are field labels and the values can
                   be None, string, integer, float, boolean, an array/list of
                   strings, integers, floats or booleans.

       The file format is similar to the Gaussian fchk format, but has the extra
       feature that the shapes of the arrays are also stored.
    '''
    with open(filename, 'w') as f:
        for key, value in sorted(data.items()):
            if not isinstance(key, str):
                raise TypeError('The keys must be strings.')
            if len(key) > 40:
                raise ValueError('Key strings cannot be longer than 40 characters.')
            if '\n' in key:
                raise ValueError('Key strings cannot contain newlines.')
            if isinstance(value, str):
                if len(value) > 256:
                    raise ValueError('Only small strings are supported (256 chars).')
                if '\n' in value:
                    raise ValueError('The string cannot contain new lines.')
                print('%40s  kind=str   %s' % (key.ljust(40), value), file=f)
            elif isinstance(value, bool):
                print('%40s  kind=bln   %s' % (key.ljust(40), value), file=f)
            elif isinstance(value, (int, np.integer)):
                print('%40s  kind=int   %i' % (key.ljust(40), value), file=f)
            elif isinstance(value, float):
                print('%40s  kind=flt   %22.15e' % (key.ljust(40), value), file=f)
            elif isinstance(value, np.ndarray) or isinstance(value, list) or \
                 isinstance(value, tuple):
                if isinstance(value, list) or isinstance(value, tuple):
                    value = np.array(value)
                if value.dtype.fields is not None:
                    raise TypeError('Arrays with fields are not supported.')
                shape_str = ','.join(str(i) for i in value.shape)
                if issubclass(value.dtype.type, (str, np.unicode, np.bytes_)):
                    value = value.astype(np.unicode)
                    for cell in value.flat:
                        if len(cell) >= 22:
                            raise ValueError('In case of string arrays, a string may contain at most 21 characters.')
                        if ' ' in cell or '\n' in cell:
                            raise ValueError('In case of string arrays, a string may not contain spaces or new lines.')
                    print('%40s  kind=strar %s' % (key.ljust(40), shape_str), file=f)
                    format_str = '%22s'
                elif issubclass(value.dtype.type, np.integer):
                    print('%40s  kind=intar %s' % (key.ljust(40), shape_str), file=f)
                    format_str = '%22i'
                elif issubclass(value.dtype.type, np.bool_):
                    print('%40s  kind=blnar %s' % (key.ljust(40), shape_str), file=f)
                    format_str = '%22s'
                elif issubclass(value.dtype.type, float):
                    print('%40s  kind=fltar %s' % (key.ljust(40), shape_str), file=f)
                    format_str = '%22.15e'
                else:
                    raise TypeError('Numpy array type %s not supported.' % value.dtype.type)
                short_len = 4
                short = []
                for x in value.ravel():
                    short.append(x)
                    if len(short) == 4:
                        print(' '.join(format_str  % s for s in short), file=f)
                        short = []
                if len(short) > 0:
                    print(' '.join(format_str  % s for s in short), file=f)
            elif value is None:
                print('%40s  kind=none   None' % key.ljust(40), file=f)
            else:
                raise TypeError('Type %s not supported.' % type(value))