def calcemissions(rates: xarray.DataArray, sim) -> Tuple[xarray.DataArray, np.ndarray, np.ndarray]:
    if not sim.reacreq:
        return 0., 0., 0.

    ver = None
    lamb = None
    br = None
    """
    Franck-Condon factor
    http://chemistry.illinoisstate.edu/standard/che460/handouts/460-Feb28lec-S13.pdf
    http://assign3.chem.usyd.edu.au/spectroscopy/index.php
    """
# %% METASTABLE
    if 'metastable' in sim.reacreq:
        ver, lamb, br = getMetastable(rates, ver, lamb, br, sim.reactionfn)
# %% PROMPT ATOMIC OXYGEN EMISSIONS
    if 'atomic' in sim.reacreq:
        ver, lamb, br = getAtomic(rates, ver, lamb, br, sim.reactionfn)
# %% N2 1N EMISSIONS
    if 'n21ng' in sim.reacreq:
        ver, lamb, br = getN21NG(rates, ver, lamb, br, sim.reactionfn)
# %% N2+ Meinel band
    if 'n2meinel' in sim.reacreq:
        ver, lamb, br = getN2meinel(rates, ver, lamb, br, sim.reactionfn)
# %% N2 2P (after Vallance Jones, 1974)
    if 'n22pg' in sim.reacreq:
        ver, lamb, br = getN22PG(rates, ver, lamb, br, sim.reactionfn)
# %% N2 1P
    if 'n21pg' in sim.reacreq:
        ver, lamb, br = getN21PG(rates, ver, lamb, br, sim.reactionfn)
# %% Remove NaN wavelength entries
    if ver is None:
        raise ValueError('you have not selected any reactions to generate VER')
# %% sort by wavelength, eliminate NaN
    lamb, ver, br = sortelimlambda(lamb, ver, br)
# %% assemble output
    dfver = xarray.DataArray(data=ver, coords=[('alt_km', rates.alt_km),
                                               ('wavelength_nm', lamb)])

    return dfver, ver, br