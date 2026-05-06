def _create_filenames(filename_schema, feed_type):
    """
    Returns a dictionary of beam filename pairs,
    keyed on correlation,from the cartesian product
    of correlations and real, imaginary pairs

    Given 'beam_$(corr)_$(reim).fits' returns:
    {
      'xx' : ('beam_xx_re.fits', 'beam_xx_im.fits'),
      'xy' : ('beam_xy_re.fits', 'beam_xy_im.fits'),
      ...
      'yy' : ('beam_yy_re.fits', 'beam_yy_im.fits'),
    }

    Given 'beam_$(CORR)_$(REIM).fits' returns:
    {
      'xx' : ('beam_XX_RE.fits', 'beam_XX_IM.fits'),
      'xy' : ('beam_XY_RE.fits', 'beam_XY_IM.fits'),
      ...
      'yy' : ('beam_YY_RE.fits', 'beam_YY_IM.fits'),
    }

    """
    template = FitsFilenameTemplate(filename_schema)

    def _re_im_filenames(corr, template):
        try:
            return tuple(template.substitute(
                corr=corr.lower(), CORR=corr.upper(),
                reim=ri.lower(), REIM=ri.upper())
                    for ri in REIM)
        except KeyError:
            raise ValueError("Invalid filename schema '%s'. "
                            "FITS Beam filename schemas "
                            "must follow forms such as "
                            "'beam_$(corr)_$(reim).fits' or "
                            "'beam_$(CORR)_$(REIM).fits." % filename_schema)

    if feed_type == 'linear':
        CORRELATIONS = LINEAR_CORRELATIONS
    elif feed_type == 'circular':
        CORRELATIONS = CIRCULAR_CORRELATIONS
    else:
        raise ValueError("Invalid feed_type '{}'. "
            "Should be 'linear' or 'circular'")

    return collections.OrderedDict(
        (c, _re_im_filenames(c, template))
        for c in CORRELATIONS)