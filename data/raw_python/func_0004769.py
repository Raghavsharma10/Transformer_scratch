def load_ref_spectra():
    """ Pull out wl, flux, ivar from files of training spectra """
    data_dir = "/Users/annaho/Data/AAOmega/ref_spectra"
    # Load the files & count the number of training objects
    ff = glob.glob("%s/*.txt" %data_dir)
    nstars = len(ff)
    print("We have %s training objects" %nstars)
    
    # Read the first file to get the wavelength array
    f = ff[0]
    data = Table.read(f, format="ascii.fast_no_header")
    wl = data['col1']
    npix = len(wl)
    print("We have %s pixels" %npix)

    tr_flux = np.zeros((nstars,npix))
    tr_ivar = np.zeros(tr_flux.shape)

    for i,f in enumerate(ff):
        data = Table.read(f, format="ascii.fast_no_header")
        flux = data['col2']
        tr_flux[i,:] = flux
        sigma = data['col3']
        tr_ivar[i,:] = 1.0 / sigma**2

    return np.array(ff), wl, tr_flux, tr_ivar