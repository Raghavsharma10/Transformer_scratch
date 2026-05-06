def readASNTable(fname, output=None, prodonly=False):
    """
    Given a fits filename repesenting an association table reads in the table as a
    dictionary which can be used by pydrizzle and multidrizzle.

    An association table is a FITS binary table with 2 required columns: 'MEMNAME',
    'MEMTYPE'. It checks 'MEMPRSNT' column and removes all files for which its value is 'no'.

    Parameters
    ----------
    fname : str
        name of association table
    output : str
        name of output product - if not specified by the user,
        the first PROD-DTH name is used if present,
        if not, the first PROD-RPT name is used if present,
        if not, the rootname of the input association table is used.
    prodonly : bool
        what files should be considered as input
        if True - select only MEMTYPE=PROD* as input
        if False - select only MEMTYPE=EXP as input

    Returns
    -------
    asndict : dict
        A dictionary-like object with all the association information.

    Examples
    --------
    An association table can be read from a file using the following commands::

    >>> from stsci.tools import asnutil
    >>> asntab = asnutil.readASNTable('j8bt06010_shifts_asn.fits', prodonly=False)  # doctest: +SKIP

    The `asntab` object can now be passed to other code to provide relationships
    between input and output images defined by the association table.

    """

    try:
        f = fits.open(fu.osfn(fname))
    except:
        raise IOError("Can't open file %s\n" % fname)

    colnames = f[1].data.names
    try:
        colunits = f[1].data.units
    except AttributeError: pass

    hdr = f[0].header

    if 'MEMNAME' not in colnames or 'MEMTYPE' not in colnames:
        msg = 'Association table incomplete: required column(s) MEMNAME/MEMTYPE NOT found!'
        raise ValueError(msg)

    d = {}
    for n in colnames:
        d[n]=f[1].data.field(n)
    f.close()

    valid_input = d['MEMPRSNT'].copy()
    memtype = d['MEMTYPE'].copy()
    prod_dth = (memtype.find('PROD-DTH')==0).nonzero()[0]
    prod_rpt = (memtype.find('PROD-RPT')==0).nonzero()[0]
    prod_crj = (memtype.find('PROD-CRJ')==0).nonzero()[0]

    # set output name
    if output is None:
        if prod_dth:
            output = d['MEMNAME'][prod_dth[0]]
        elif prod_rpt:
            output = d['MEMNAME'][prod_rpt[0]]
        elif prod_crj:
            output = d['MEMNAME'][prod_crj[0]]
        else:
            output = fname.split('_')[0]

    if prodonly:
        input = d['MEMTYPE'].find('PROD')==0
        if prod_dth:
            input[prod_dth] = False
    else:
        input = (d['MEMTYPE'].find('EXP')==0)
    valid_input *= input

    for k in d:
        d[k] = d[k][valid_input]

    infiles = list(d['MEMNAME'].lower())
    if not infiles:
        print("No valid input specified")
        return None

    if ('XOFFSET' in colnames and d['XOFFSET'].any()) or ('YOFFSET' in colnames and d['YOFFSET'].any()):
        abshift = True
        dshift = False
        try:
            units=colunits[colnames.index('XOFFSET')]
        except: units='pixels'
        xshifts = list(d['XOFFSET'])
        yshifts = list(d['YOFFSET'])
    elif ('XDELTA' in colnames and d['XDELTA'].any()) or  ('YDELTA' in colnames and d['YDELTA'].any()):
        abshift = False
        dshift = True
        try:
            units=colunits[colnames.index('XDELTA')]
        except: units='pixels'
        xshifts = list(d['XDELTA'])
        yshifts = list(d['YDELTA'])
    else:
        abshift = False
        dshift = False
    members = {}

    if not abshift and not dshift:
        asndict = ASNTable(infiles,output=output)
        asndict.create()
        return asndict
    else:
        try:
            refimage = hdr['refimage']
        except KeyError: refimage = None
        try:
            frame = hdr['shframe']
        except KeyError: frame = 'input'
        if 'ROTATION' in colnames:
            rots = list(d['ROTATION'])
        if 'SCALE' in colnames:
            scales = list(d['SCALE'])

        for r in range(len(infiles)):
            row = r
            xshift = xshifts[r]
            yshift = yshifts[r]
            if rots: rot = rots[r]
            if scales: scale = scales[r]
            members[infiles[r]] = ASNMember(row=row, dshift=dshift, abshift=abshift, rot=rot, xshift=xshift,
                                      yshift=yshift, scale=scale, refimage=refimage, shift_frame=frame,
                                      shift_units=units)


        asndict= ASNTable(infiles, output=output)
        asndict.create()
        asndict['members'].update(members)
        return asndict