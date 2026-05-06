def convert(input, width=132, output=None, keep=False):

    """Input ASCII trailer file "input" will be read.

    The contents will then be written out to a FITS file in the same format
    as used by 'stwfits' from IRAF.

    Parameters
    ===========
    input : str
        Filename of input ASCII trailer file

    width : int
        Number of characters wide to use for defining output FITS column
        [Default: 132]

    output : str
        Filename to use for writing out converted FITS trailer file
        If None, input filename will be converted from *.tra -> *_trl.fits
        [Default: None]

    keep : bool
        Specifies whether or not to keep any previously written FITS files
        [Default: False]

    """
    # open input trailer file
    trl = open(input)

    # process all lines
    lines = np.array([i for text in trl.readlines() for i in textwrap.wrap(text,width=width)])

    # close ASCII trailer file now that we have processed all the lines
    trl.close()

    if output is None:
        # create fits file
        rootname,suffix = os.path.splitext(input)
        s = suffix[1:].replace('ra','rl')
        fitsname = "{}_{}{}fits".format(rootname,s,os.path.extsep)
    else:
        fitsname = output
    full_name = os.path.abspath(os.path.join(os.path.curdir,fitsname))

    old_file = os.path.exists(full_name)
    if old_file:
        if keep:
            print("ERROR: Trailer file already written out as: {}".format(full_name))
            raise IOError
        else:
            os.remove(full_name)

    # Build FITS table and write it out
    line_fmt = "{}A".format(width)
    tbhdu = fits.BinTableHDU.from_columns([fits.Column(name='TEXT_FILE',format=line_fmt,array=lines)])
    tbhdu.writeto(fitsname)

    print("Created output FITS filename for trailer:{}    {}".format(os.linesep,full_name))

    os.remove(input)