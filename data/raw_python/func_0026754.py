def set_pd_mag_from_flux_density(photodict,
                                 fd='',
                                 efd='',
                                 lefd='',
                                 uefd='',
                                 sig=DEFAULT_UL_SIGMA):
    """Set photometry dictionary from a flux density measurement.

    `fd` is assumed to be in microjanskys.
    """
    with localcontext() as ctx:
        if lefd == '' or uefd == '':
            lefd = efd
            uefd = efd
        prec = max(
            get_sig_digits(str(fd), strip_zeroes=False),
            get_sig_digits(str(lefd), strip_zeroes=False),
            get_sig_digits(str(uefd), strip_zeroes=False)) + 1
        ctx.prec = prec
        dlefd = Decimal(str(lefd))
        duefd = Decimal(str(uefd))
        if fd != '':
            dfd = Decimal(str(fd))
        dsig = Decimal(str(sig))
        if fd == '' or float(fd) < DEFAULT_UL_SIGMA * float(uefd):
            photodict[PHOTOMETRY.UPPER_LIMIT] = True
            photodict[PHOTOMETRY.UPPER_LIMIT_SIGMA] = str(sig)
            photodict[PHOTOMETRY.MAGNITUDE] = str(Decimal('23.9') - D25 * (
                dsig * duefd).log10())
            if fd:
                photodict[PHOTOMETRY.E_UPPER_MAGNITUDE] = str(D25 * (
                    (dfd + duefd).log10() - dfd.log10()))
        else:
            photodict[PHOTOMETRY.MAGNITUDE] = str(Decimal('23.9') - D25 *
                                                  dfd.log10())
            photodict[PHOTOMETRY.E_UPPER_MAGNITUDE] = str(D25 * (
                (dfd + duefd).log10() - dfd.log10()))
            photodict[PHOTOMETRY.E_LOWER_MAGNITUDE] = str(D25 * (
                dfd.log10() - (dfd - dlefd).log10()))