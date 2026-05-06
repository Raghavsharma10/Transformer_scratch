def readBerSANS(filename):
    """Read a header from a SANS file (produced usually by BerSANS)"""
    hed = {'Comment': ''}
    translate = {'Lambda': 'Wavelength',
                 'Title': 'Owner',
                 'SampleName': 'Title',
                 'BeamcenterX': 'BeamPosY',
                 'BeamcenterY': 'BeamPosX',
                 'Time': 'MeasTime',
                 'TotalTime': 'MeasTime',
                 'Moni1': 'Monitor',
                 'Moni2': 'Monitor',
                 'Moni': 'Monitor',
                 'Transmission': 'Transm',
                 }
    with open(filename, 'rt') as f:
        comment_next = False
        for l in f:
            l = l.strip()
            if comment_next:
                hed['Comment'] = hed['Comment'] + '\n' + l
                comment_next = False
            elif l.startswith('%Counts'):
                break
            elif l.startswith('%Comment'):
                comment_next = True
            elif l.startswith('%'):
                continue
            elif l.split('=', 1)[0] in translate:
                hed[translate[l.split('=', 1)[0]]] = misc.parse_number(
                    l.split('=', 1)[1])
            else:
                try:
                    hed[l.split('=', 1)[0]] = misc.parse_number(
                        l.split('=', 1)[1])
                except IndexError:
                    print(l.split('=', 1))
    if 'FileName' in hed:
        m = re.match('D(\d+)\.(\d+)', hed['FileName'])
        if m is not None:
            hed['FSN'] = int(m.groups()[0])
            hed['suffix'] = int(m.groups()[1])
    if 'FileDate' in hed:
        hed['Date'] = dateutil.parser.parse(hed['FileDate'])
    if 'FileTime' in hed:
        hed['Date'] = datetime.datetime.combine(
            hed['Date'].date(), dateutil.parser.parse(hed['FileTime']).time())
    hed['__Origin__'] = 'BerSANS'
    if 'SD' in hed:
        hed['Dist'] = hed['SD'] * 1000
    if hed['Comment'].startswith('\n'):
        hed['Comment'] = hed['Comment'][1:]
    hed['__particle__'] = 'neutron'
    hed['Wavelength'] *= 10  # convert from nanometres to Angstroems
    return hed