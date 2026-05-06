def etau_madau(wave, z, **kwargs):
    """Madau 1995 extinction for a galaxy at given redshift.
    This is the Lyman-alpha prescription from the photo-z code BPZ.

    The Lyman-alpha forest approximately has an effective
    "throughput" which is a function of redshift and
    rest-frame wavelength.
    One would multiply the SEDs by this factor before
    passing it through an instrument filter.

    This approximation is from Footnote 3 of
    :ref:`Madau et al. (1995) <synphot-ref-madau1995>`.
    This is claimed accurate to 5%.
    The scatter in this factor (due to different lines of sight)
    is huge, as shown in Madau's Fig. 3 (top panel);
    The figure's bottom panel shows a redshifted version of the
    "exact" prescription.

    Parameters
    ----------
    wave : array-like or `~astropy.units.quantity.Quantity`
        Redshifted wavelength values.
        Non-redshifted wavelength is ``wave / (1 + z)``.

    z : number
        Redshift.

    kwargs : dict
        Equivalencies for unit conversion, see
        :func:`~synphot.units.validate_quantity`.

    Returns
    -------
    extcurve : `ExtinctionCurve`
        Extinction curve to apply to the redshifted spectrum.

    """
    if not isinstance(z, numbers.Real):
        raise exceptions.SynphotError(
            'Redshift must be a real scalar number.')

    if np.isscalar(wave) or len(wave) <= 1:
        raise exceptions.SynphotError('Wavelength has too few data points')

    wave = units.validate_quantity(wave, u.AA, **kwargs).value

    ll = 912.0
    c = np.array([3.6e-3, 1.7e-3, 1.2e-3, 9.3e-4])
    el = np.array([1216, 1026, 973, 950], dtype=np.float)  # noqa
    tau = np.zeros_like(wave, dtype=np.float)
    xe = 1.0 + z

    # Lyman series
    for i in range(len(el)):
        tau = np.where(wave <= el[i] * xe,
                       tau + c[i] * (wave / el[i]) ** 3.46,
                       tau)

    # Photoelectric absorption
    xc = wave / ll
    xc3 = xc ** 3
    tau = np.where(wave <= ll * xe,
                   (tau + 0.25 * xc3 * (xe ** 0.46 - xc ** 0.46) +
                    9.4 * xc ** 1.5 * (xe ** 0.18 - xc ** 0.18) -
                    0.7 * xc3 * (xc ** (-1.32) - xe ** (-1.32)) -
                    0.023 * (xe ** 1.68 - xc ** 1.68)),
                   tau)

    thru = np.where(tau > 700., 0., np.exp(-tau))
    meta = {'descrip': 'Madau 1995 extinction for z={0}'.format(z)}
    return ExtinctionCurve(ExtinctionModel1D, points=wave, lookup_table=thru,
                           meta=meta)