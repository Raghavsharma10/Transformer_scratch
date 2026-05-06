def readB1header(filename):
    """Read beamline B1 (HASYLAB, Hamburg) header data

    Input
    -----
    filename: string
        the file name. If ends with ``.gz``, it is fed through a ``gunzip``
        filter

    Output
    ------
    A header dictionary.

    Examples
    --------
    read header data from 'ORG000123.DAT'::

        header=readB1header('ORG00123.DAT')
    """
    # Planck's constant times speed of light: incorrect
    # constant in the old program on hasjusi1, which was
    # taken over by the measurement program, to keep
    # compatibility with that.
    hed = {}
    if libconfig.LENGTH_UNIT == 'A':
        jusifaHC = 12396.4
    elif libconfig.LENGTH_UNIT == 'nm':
        jusifaHC = 1239.64
    else:
        raise NotImplementedError(
            'Invalid length unit: ' + str(libconfig.LENGTH_UNIT))

    if filename.upper().endswith('.GZ'):
        fid = gzip.GzipFile(filename, 'r')
    else:
        fid = open(filename, 'rt')
    lines = fid.readlines()
    fid.close()
    hed['FSN'] = int(lines[0].strip())
    hed['Hour'] = int(lines[17].strip())
    hed['Minutes'] = int(lines[18].strip())
    hed['Month'] = int(lines[19].strip())
    hed['Day'] = int(lines[20].strip())
    hed['Year'] = int(lines[21].strip()) + 2000
    hed['FSNref1'] = int(lines[23].strip())
    hed['FSNdc'] = int(lines[24].strip())
    hed['FSNsensitivity'] = int(lines[25].strip())
    hed['FSNempty'] = int(lines[26].strip())
    hed['FSNref2'] = int(lines[27].strip())
    hed['Monitor'] = float(lines[31].strip())
    hed['Anode'] = float(lines[32].strip())
    hed['MeasTime'] = float(lines[33].strip())
    hed['Temperature'] = float(lines[34].strip())
    hed['BeamPosX'] = float(lines[36].strip())
    hed['BeamPosY'] = float(lines[37].strip())
    hed['Transm'] = float(lines[41].strip())
    hed['Wavelength'] = float(lines[43].strip())
    hed['Energy'] = jusifaHC / hed['Wavelength']
    hed['Dist'] = float(lines[46].strip())
    hed['XPixel'] = 1 / float(lines[49].strip())
    hed['YPixel'] = 1 / float(lines[50].strip())
    hed['Title'] = lines[53].strip().replace(' ', '_').replace('-', '_')
    hed['MonitorDORIS'] = float(lines[56].strip())  # aka. DORIS counter
    hed['Owner'] = lines[57].strip()
    hed['RotXSample'] = float(lines[59].strip())
    hed['RotYSample'] = float(lines[60].strip())
    hed['PosSample'] = float(lines[61].strip())
    hed['DetPosX'] = float(lines[62].strip())
    hed['DetPosY'] = float(lines[63].strip())
    hed['MonitorPIEZO'] = float(lines[64].strip())  # aka. PIEZO counter
    hed['BeamsizeX'] = float(lines[66].strip())
    hed['BeamsizeY'] = float(lines[67].strip())
    hed['PosRef'] = float(lines[70].strip())
    hed['Monochromator1Rot'] = float(lines[77].strip())
    hed['Monochromator2Rot'] = float(lines[78].strip())
    hed['Heidenhain1'] = float(lines[79].strip())
    hed['Heidenhain2'] = float(lines[80].strip())
    hed['Current1'] = float(lines[81].strip())
    hed['Current2'] = float(lines[82].strip())
    hed['Detector'] = 'Unknown'
    hed['PixelSize'] = (hed['XPixel'] + hed['YPixel']) / 2.0

    hed['AnodeError'] = math.sqrt(hed['Anode'])
    hed['TransmError'] = 0
    hed['MonitorError'] = math.sqrt(hed['Monitor'])
    hed['MonitorPIEZOError'] = math.sqrt(hed['MonitorPIEZO'])
    hed['MonitorDORISError'] = math.sqrt(hed['MonitorDORIS'])
    hed['Date'] = datetime.datetime(
        hed['Year'], hed['Month'], hed['Day'], hed['Hour'], hed['Minutes'])
    hed['__Origin__'] = 'B1 original'
    hed['__particle__'] = 'photon'
    return hed