def SNRPlanet(SNRStar, starPlanetFlux, Nobs, pixPerbin, NVisits=1):
    r""" Calculate the Signal to Noise Ratio of the planet atmosphere

    .. math::
        \text{SNR}_\text{planet} = \text{SNR}_\text{star} \times \Delta F \times
        \sqrt{N_\text{obs}}
        \times \sqrt{N_\text{pixPerbin}} \times \sqrt{N_\text{visits}}

    Where :math:`\text{SNR}_\star` SNR of the star detection, :math:`\Delta F`
    ratio of the terminator to the star, :math:`N_\text{obs}` number of
    exposures per visit, :math:`N_\text{pixPerBin}` number of pixels per
    wavelength bin, :math:`N_\text{visits}` number of visits.

    :return:
    """

    SNRplanet = SNRStar * starPlanetFlux * \
        sqrt(Nobs) * sqrt(pixPerbin) * sqrt(NVisits)

    return SNRplanet