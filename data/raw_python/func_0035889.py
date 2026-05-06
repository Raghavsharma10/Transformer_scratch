def delta13c_santrock(r45sam, r46sam, d13cstd, r45std, r46std,
                      ks='Santrock', d18ostd=23.5):
    """
    Given the measured isotope signals of a sample and a
    standard and the delta-13C of that standard, calculate
    the delta-13C of the sample.

    Algorithm from Santrock, Studley & Hayes 1985 Anal. Chem.
    """
    k = delta13c_constants()[ks]

    # function for calculating 17R from 18R
    def c17(r):
        return k['K'] * r ** k['A']
    rcpdb, rosmow = k['S13'], k['S18']

    # known delta values for the ref peak
    r13std = (d13cstd / 1000. + 1) * rcpdb
    r18std = (d18ostd / 1000. + 1) * rosmow

    # determine the correction factors
    c45 = r13std + 2 * c17(r18std)
    c46 = c17(r18std) ** 2 + 2 * r13std * c17(r18std) + 2 * r18std

    # correct the voltage ratios to ion ratios
    r45 = (r45sam / r45std) * c45
    r46 = (r46sam / r46std) * c46

    def rf(r18):
        return -3 * c17(r18) ** 2 + 2 * r45 * c17(r18) + 2 * r18 - r46
    # r18 = scipy.optimize.root(rf, r18std).x[0]  # use with scipy 0.11.0
    r18 = fsolve(rf, r18std)[0]
    r13 = r45 - 2 * c17(r18)
    return 1000 * (r13 / rcpdb - 1)