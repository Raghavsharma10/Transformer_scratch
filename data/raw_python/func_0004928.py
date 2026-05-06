def readcbf(name, load_header=False, load_data=True, for_nexus=False):
    """Read a cbf (crystallographic binary format) file from a Dectris PILATUS
    detector.

    Inputs
    ------
    name: string
        the file name
    load_header: bool
        if the header data is to be loaded.
    load_data: bool
        if the binary data is to be loaded.
    for_nexus: bool
        if the array should be opened with NeXus ordering.

    Output
    ------
    a numpy array of the scattering data

    Notes
    -----
    currently only Little endian, "signed 32-bit integer" type and
    byte-offset compressed data are accepted.
    """
    with open(name, 'rb') as f:
        cbfbin = f.read()
    datastart = cbfbin.find(b'\x0c\x1a\x04\xd5') + 4
    hed = [x.strip() for x in cbfbin[:datastart].split(b'\n')]
    header = {}
    readingmode = None
    for i in range(len(hed)):
        if not hed[i]:
            # skip empty header lines
            continue
        elif hed[i] == b';':
            continue
        elif hed[i].startswith(b'_array_data.header_convention'):
            header['CBF_header_convention'] = str(hed[i][
                len(b'_array_data.header_convention'):].strip().replace(b'"', b''), encoding='utf-8')
        elif hed[i].startswith(b'_array_data.header_contents'):
            readingmode = 'PilatusHeader'
        elif hed[i].startswith(b'_array_data.data'):
            readingmode = 'CIFHeader'
        elif readingmode == 'PilatusHeader':
            if not hed[i].startswith(b'#'):
                continue
            line = hed[i].strip()[1:].strip()
            try:
                # try to interpret the line as the date.
                header['CBF_Date'] = dateutil.parser.parse(line)
                header['Date'] = header['CBF_Date']
                continue
            except (ValueError, TypeError):
                # eat exception: if we cannot parse this line as a date, try
                # another format.
                pass
            treated = False
            for sep in (b':', b'='):
                if treated:
                    continue
                if line.count(sep) == 1:
                    name, value = tuple(x.strip() for x in line.split(sep, 1))
                    try:
                        m = re.match(
                            b'^(?P<number>-?(\d+(.\d+)?(e-?\d+)?))\s+(?P<unit>m|s|counts|eV)$', value).groupdict()
                        value = float(m['number'])
                        m['unit'] = str(m['unit'], encoding='utf-8')
                    except AttributeError:
                        # the regex did not match the string, thus re.match()
                        # returned None.
                        pass
                    header[str(name, 'utf-8')] = value
                    treated = True
            if treated:
                continue
            if line.startswith(b'Pixel_size'):
                header['XPixel'], header['YPixel'] = tuple(
                    [float(a.strip().split(b' ')[0]) * 1000 for a in line[len(b'Pixel_size'):].split(b'x')])
            else:
                try:
                    m = re.match(
                        b'^(?P<label>[a-zA-Z0-9,_\.\-!\?\ ]*?)\s+(?P<number>-?(\d+(.\d+)?(e-?\d+)?))\s+(?P<unit>m|s|counts|eV)$', line).groupdict()
                except AttributeError:
                    pass
                else:
                    m['label'] = str(m['label'], 'utf-8')
                    m['unit'] = str(m['unit'], encoding='utf-8')
                    if m['unit'] == b'counts':
                        header[m['label']] = int(m['number'])
                    else:
                        header[m['label']] = float(m['number'])
                    if 'sensor' in m['label'] and 'thickness' in m['label']:
                        header[m['label']] *= 1e6
        elif readingmode == 'CIFHeader':
            line = hed[i]
            for sep in (b':', b'='):
                if line.count(sep) == 1:
                    label, content = tuple(x.strip()
                                           for x in line.split(sep, 1))
                    if b'"' in content:
                        content = content.replace(b'"', b'')
                    try:
                        content = int(content)
                    except ValueError:
                        content = str(content, encoding='utf-8')
                    header['CBF_' + str(label, encoding='utf-8')] = content

        else:
            pass
    ret = []
    if load_data:
        if header['CBF_X-Binary-Element-Type'] != 'signed 32-bit integer':
            raise NotImplementedError(
                'element type is not "signed 32-bit integer" in CBF, but %s.' % header['CBF_X-Binary-Element-Type'])
        if header['CBF_conversions'] != 'x-CBF_BYTE_OFFSET':
            raise NotImplementedError(
                'compression is not "x-CBF_BYTE_OFFSET" in CBF!')
        dim1 = header['CBF_X-Binary-Size-Fastest-Dimension']
        dim2 = header['CBF_X-Binary-Size-Second-Dimension']
        nbytes = header['CBF_X-Binary-Size']
        cbfdata = cbfdecompress(
            bytearray(cbfbin[datastart:datastart + nbytes]), dim1, dim2, for_nexus)
        ret.append(cbfdata)
    if load_header:
        ret.append(header)
    return tuple(ret)