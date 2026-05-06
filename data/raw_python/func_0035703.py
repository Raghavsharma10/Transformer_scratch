def read_reg_file(f, foff=0x2D):
    """
    Given a file handle for an old-style Agilent *.REG file, this
    will parse that file into a dictonary of key/value pairs
    (including any tables that are in the *.REG file, which will
    be parsed into lists of lists).
    """
    # convenience function for reading in data
    def rd(st):
        return struct.unpack(st, f.read(struct.calcsize(st)))

    f.seek(0x19)
    if f.read(1) != b'A':
        # raise TypeError("Version of REG file is too new.")
        return {}

    f.seek(foff)
    nrecs = rd('<I')[0]  # TODO: should be '<H'
    rec_tab = [rd('<HHIII') for n in range(nrecs)]

    names = {}
    f.seek(foff + 20 * nrecs + 4)
    for r in rec_tab:
        d = f.read(r[2])
        if r[1] == 1539:  # '0306'
            # this is part of the linked list too, but contains a
            # reference to a table
            cd = struct.unpack('<HIII21sI', d)
            names[cd[5]] = cd[4].decode('iso8859').strip('\x00')
            # except:
            #     pass
        elif r[1] == 32769 or r[1] == 32771:  # b'0180' or b'0380'
            names[r[4]] = d[:-1].decode('iso8859')
        elif r[1] == 32774:  # b'0680'
            # this is a string that is referenced elsewhere (in a table)
            names[r[4]] = d[2:-1].decode('iso8859')
        elif r[1] == 32770:  # b'0280'
            # this is just a flattened numeric array
            names[r[4]] = np.frombuffer(d, dtype=np.uint32, offset=4)

    data = {}
    f.seek(foff + 20 * nrecs + 4)
    for r in rec_tab:
        d = f.read(r[2])
        if r[1] == 1538:  # '0206'
            # this is part of a linked list
            if len(d) == 43:
                cd = struct.unpack('<HIII21sd', d)
                data[cd[4].decode('iso8859').strip('\x00')] = cd[5]
            else:
                pass
        elif r[1] == 1537:  # b'0106'
            # name of property
            n = d[14:30].split(b'\x00')[0].decode('iso8859')
            # with value from names
            data[n] = names.get(struct.unpack('<I', d[35:39])[0], '')
        elif r[1] == 1793:  # b'0107'
            # this is a table of values
            nrow = struct.unpack('<H', d[4:6])[0]
            ncol = struct.unpack('<H', d[16:18])[0]
            if ncol != 0:
                cols = [struct.unpack('<16sHHHHHI', d[20 + 30 * i:50 + 30 * i])
                        for i in range(ncol)]
                colnames = [c[0].split(b'\x00')[0].decode('iso8859')
                            for c in cols]
                # TODO: type 2 is not a constant size? 31, 17
                rty2sty = {1: 'H', 3: 'I', 4: 'f', 5: 'H',
                           7: 'H', 8: 'd', 11: 'H', 12: 'H',
                           13: 'I', 14: 'I', 16: 'H'}
                coltype = '<' + ''.join([rty2sty.get(c[3], str(c[2]) + 's')
                                         for c in cols])
                lencol = struct.calcsize(coltype)
                tab = []
                for i in reversed(range(2, nrow + 2)):
                    rawrow = struct.unpack(coltype,
                                           d[-i * lencol: (1 - i) * lencol])
                    row = []
                    for j, p in enumerate(rawrow):
                        if cols[j][3] == 3:
                            row.append(names.get(p, str(p)))
                        else:
                            row.append(p)
                    tab.append(row)
                data[names[r[4]]] = [colnames, tab]
        elif r[1] == 1281 or r[1] == 1283:  # b'0105' or b'0305'
            fm = '<HHBIIhIdII12shIddQQB8sII12shIddQQB8s'
            m = struct.unpack(fm, d)
            nrecs = m[4]  # number of points in table

            # x_units = names.get(m[8], '')
            x_arr = m[14] * names.get(m[9], np.arange(nrecs - 1))
            y_arr = m[25] * names.get(m[20])
            y_units = names.get(m[19], '')
            if y_units == 'bar':
                y_arr *= 0.1  # convert to MPa
            # TODO: what to call this?
            data['Trace'] = Trace(y_arr, x_arr, name='')
        # elif r[1] == 1025:  # b'0104'
        #     # lots of zeros? maybe one or two numbers?
        #     # only found in REG entries that have long 0280 records
        #     fm = '<HQQQIHHHHIIHB'
        #     m = struct.unpack(fm, d)
        #     print(m)
        #     #print(r[1], len(d), binascii.hexlify(d))
        #     pass
        # elif r[1] == 512:  # b'0002'
        #     # either points to two null pointers or two other pointers
        #     # (indicates start of linked list?)
        #     print(r[1], len(d), binascii.hexlify(d))
        # elif r[1] == 769 or r[1] == 772:  # b'0103' or b'0403'
        #     # points to 2nd, 3rd & 4th records (two 0002 records and a 0180)
        #     b = binascii.hexlify
        #     print(b(d[10:14]), b(d[14:18]), b(d[18:22]))

    return data