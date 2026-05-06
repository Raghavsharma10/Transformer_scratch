def create_header_from_telpars(telpars):
    """
    Create a list of fits header items from GTC telescope pars.

    The GTC telescope server gives a list of string describing
    FITS header items such as RA, DEC, etc.

    Arguments
    ---------
    telpars : list
        list returned by server call to getTelescopeParams
    """
    # pars is a list of strings describing tel info in FITS
    # style, each entry in the list is a different class of
    # thing (weather, telescope, instrument etc).

    # first, we munge it into a single list of strings, each one
    # describing a single item whilst also stripping whitespace
    pars = [val.strip() for val in (';').join(telpars).split(';')
            if val.strip() != '']

    # apply parse_hstring to everything in pars
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fits.verify.VerifyWarning)
        hdr = fits.Header(map(parse_hstring, pars))

    return hdr