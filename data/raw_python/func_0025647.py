def readgeis(input):

    """Input GEIS files "input" will be read and a HDUList object will
       be returned.

       The user can use the writeto method to write the HDUList object to
       a FITS file.
    """

    global dat
    cardLen = fits.Card.length

    # input file(s) must be of the form *.??h and *.??d
    if input[-1] != 'h' or input[-4] != '.':
        raise "Illegal input GEIS file name %s" % input

    data_file = input[:-1]+'d'

    _os = sys.platform
    if _os[:5] == 'linux' or _os[:5] == 'win32' or _os[:5] == 'sunos' or _os[:3] == 'osf' or _os[:6] == 'darwin':
        bytes_per_line = cardLen+1
    else:
        raise "Platform %s is not supported (yet)." % _os

    geis_fmt = {'REAL':'f', 'INTEGER':'i', 'LOGICAL':'i','CHARACTER':'S'}
    end_card = 'END'+' '* (cardLen-3)

    # open input file
    im = open(input)

    # Generate the primary HDU
    cards = []
    while 1:
        line = im.read(bytes_per_line)[:cardLen]
        line = line[:8].upper() + line[8:]
        if line == end_card:
            break
        cards.append(fits.Card.fromstring(line))

    phdr = fits.Header(cards)
    im.close()

    _naxis0 = phdr.get('NAXIS', 0)
    _naxis = [phdr['NAXIS'+str(j)] for j in range(1, _naxis0+1)]
    _naxis.insert(0, _naxis0)
    _bitpix = phdr['BITPIX']
    _psize = phdr['PSIZE']
    if phdr['DATATYPE'][:4] == 'REAL':
        _bitpix = -_bitpix
    if _naxis0 > 0:
        size = reduce(lambda x,y:x*y, _naxis[1:])
        data_size = abs(_bitpix) * size // 8
    else:
        data_size = 0
    group_size = data_size + _psize // 8

    # decode the group parameter definitions,
    # group parameters will become extension header
    groups = phdr['GROUPS']
    gcount = phdr['GCOUNT']
    pcount = phdr['PCOUNT']

    formats = []
    bools = []
    floats = []
    _range = range(1, pcount+1)
    key = [phdr['PTYPE'+str(j)] for j in _range]
    comm = [phdr.cards['PTYPE'+str(j)].comment for j in _range]

    # delete group parameter definition header keywords
    _list = ['PTYPE'+str(j) for j in _range] + \
            ['PDTYPE'+str(j) for j in _range] + \
            ['PSIZE'+str(j) for j in _range] + \
            ['DATATYPE', 'PSIZE', 'GCOUNT', 'PCOUNT', 'BSCALE', 'BZERO']

    # Construct record array formats for the group parameters
    # as interpreted from the Primary header file
    for i in range(1, pcount+1):
        ptype = key[i-1]
        pdtype = phdr['PDTYPE'+str(i)]
        star = pdtype.find('*')
        _type = pdtype[:star]
        _bytes = pdtype[star+1:]

        # collect boolean keywords since they need special attention later

        if _type == 'LOGICAL':
            bools.append(i)
        if pdtype == 'REAL*4':
            floats.append(i)

        fmt = geis_fmt[_type] + _bytes
        formats.append((ptype,fmt))

    _shape = _naxis[1:]
    _shape.reverse()
    _code = fits.BITPIX2DTYPE[_bitpix]
    _bscale = phdr.get('BSCALE', 1)
    _bzero = phdr.get('BZERO', 0)
    if phdr['DATATYPE'][:10] == 'UNSIGNED*2':
        _uint16 = 1
        _bzero = 32768
    else:
        _uint16 = 0

    # delete from the end, so it will not conflict with previous delete
    for i in range(len(phdr)-1, -1, -1):
        if phdr.cards[i].keyword in _list:
            del phdr[i]

    # clean up other primary header keywords
    phdr['SIMPLE'] = True
    phdr['BITPIX'] = 16
    phdr['GROUPS'] = False
    _after = 'NAXIS'
    if _naxis0 > 0:
        _after += str(_naxis0)
    phdr.set('EXTEND', value=True, comment="FITS dataset may contain extensions", after=_after)
    phdr.set('NEXTEND', value=gcount, comment="Number of standard extensions")

    hdulist = fits.HDUList([fits.PrimaryHDU(header=phdr, data=None)])

    # Use copy-on-write for all data types since byteswap may be needed
    # in some platforms.
    f1 = open(data_file, mode='rb')
    dat = f1.read()
#    dat = memmap(data_file, mode='c')
    hdulist.mmobject = dat

    errormsg = ""

    loc = 0
    for k in range(gcount):
        ext_dat = numpy.fromstring(dat[loc:loc+data_size], dtype=_code)
        ext_dat = ext_dat.reshape(_shape)
        if _uint16:
            ext_dat += _bzero
        # Check to see whether there are any NaN's or infs which might indicate
        # a byte-swapping problem, such as being written out on little-endian
        #   and being read in on big-endian or vice-versa.
        if _code.find('float') >= 0 and \
            (numpy.any(numpy.isnan(ext_dat)) or numpy.any(numpy.isinf(ext_dat))):
            errormsg += "===================================\n"
            errormsg += "= WARNING:                        =\n"
            errormsg += "=  Input image:                   =\n"
            errormsg += input+"[%d]\n"%(k+1)
            errormsg += "=  had floating point data values =\n"
            errormsg += "=  of NaN and/or Inf.             =\n"
            errormsg += "===================================\n"
        elif _code.find('int') >= 0:
            # Check INT data for max values
            ext_dat_frac,ext_dat_exp = numpy.frexp(ext_dat)
            if ext_dat_exp.max() == int(_bitpix) - 1:
                # Potential problems with byteswapping
                errormsg += "===================================\n"
                errormsg += "= WARNING:                        =\n"
                errormsg += "=  Input image:                   =\n"
                errormsg += input+"[%d]\n"%(k+1)
                errormsg += "=  had integer data values        =\n"
                errormsg += "=  with maximum bitvalues.        =\n"
                errormsg += "===================================\n"

        ext_hdu = fits.ImageHDU(data=ext_dat)

        rec = numpy.fromstring(dat[loc+data_size:loc+group_size], dtype=formats)

        loc += group_size

        # Create separate PyFITS Card objects for each entry in 'rec'
        for i in range(1, pcount+1):
            #val = rec.field(i-1)[0]
            val = rec[0][i-1]
            if val.dtype.kind == 'S':
                val = val.decode('ascii')

            if i in bools:
                if val:
                    val = True
                else:
                    val = False

            if i in floats:
                # use fromstring, format in Card is deprecated in pyfits 0.9
                _str = '%-8s= %20.7G / %s' % (key[i-1], val, comm[i-1])
                _card = fits.Card.fromstring(_str)
            else:
                _card = fits.Card(keyword=key[i-1], value=val, comment=comm[i-1])

            ext_hdu.header.append(_card)

        # deal with bscale/bzero
        if (_bscale != 1 or _bzero != 0):
            ext_hdu.header['BSCALE'] = _bscale
            ext_hdu.header['BZERO'] = _bzero

        hdulist.append(ext_hdu)

    if errormsg != "":
        errormsg += "===================================\n"
        errormsg += "=  This file may have been        =\n"
        errormsg += "=  written out on a platform      =\n"
        errormsg += "=  with a different byte-order.   =\n"
        errormsg += "=                                 =\n"
        errormsg += "=  Please verify that the values  =\n"
        errormsg += "=  are correct or apply the       =\n"
        errormsg += "=  '.byteswap()' method.          =\n"
        errormsg += "===================================\n"
        print(errormsg)

    f1.close()
    stsci(hdulist)
    return hdulist