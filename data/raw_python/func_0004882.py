def readB1logfile(filename):
    """Read B1 logfile (*.log)

    Inputs:
        filename: the file name

    Output: A dictionary.
    """
    dic = dict()
    # try to open. If this fails, an exception is raised
    with open(filename, 'rt', encoding='utf-8') as f:
        for l in f:
            l = l.strip()
            if l[0] in '#!%\'':
                continue  # treat this line as a comment
            try:
                # find the first tuple in _logfile_data where the first element of the
                # tuple is the starting of the line.
                ld = [ld_ for ld_ in _logfile_data if l.split(
                    ':', 1)[0].strip() == ld_[0]][0]
            except IndexError:
                # line is not recognized. We can still try to load it: find the first
                # semicolon. If found, the part of the line before it is stripped
                # from whitespaces and will be the key. The part after it is stripped
                # from whitespaces and parsed with misc.parse_number(). If no
                if ':' in l:
                    key, val = [x.strip() for x in l.split(':', 1)]
                    val = misc.parse_number(val)
                    dic[key] = val
                    try:
                        # fix the character encoding in files written by a
                        # previous version of this software.
                        dic[key] = dic[key].encode('latin2').decode('utf-8')
                    except (UnicodeDecodeError, UnicodeEncodeError, AttributeError):
                        pass
                else:
                    dic[l.strip()] = True
                continue
            try:
                reader = ld[3]
            except IndexError:
                reader = str
            rhs = l.split(':', 1)[1].strip()
            try:
                vals = reader(rhs)
            except ValueError:
                if rhs.lower() == 'none':
                    vals = None
                else:
                    raise
            if isinstance(ld[1], tuple):
                # more than one field names. The reader function should return a
                # tuple here, a value for each field.
                if len(vals) != len(ld[1]):
                    raise ValueError(
                        'Cannot read %d values from line %s in file!' % (len(ld[1]), l))
                dic.update(dict(list(zip(ld[1], vals))))
            else:
                dic[ld[1]] = vals
    dic['__Origin__'] = 'B1 log'
    dic['__particle__'] = 'photon'
    return dic