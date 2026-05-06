def delta13c_craig(r45sam, r46sam, d13cstd, r45std, r46std,
                   ks='Craig', d18ostd=23.5):
    """
    Algorithm from Craig 1957.

    From the original Craig paper, we can set up a pair of equations
    and solve for d13C and d18O simultaneously:

        d45 * r45 = r13 * d13
                  + 0.5 * r17 * d18
        d46 = r13 * ((r17**2 + r17 - r18) / a) * d13
            + 1 - 0.5 * r17 * ((r13**2 + r13 - r18) / a) * d18
        where a = r18 + r13 * r17 and b = 1 + r13 + r17
    """
    # the constants for the calculations
    # originally r13, r17, r18 = 1123.72e-5, 759.9e-6, 415.8e-5
    k = delta13c_constants()[ks]

    # TODO: not clear why need to multiply by 2?
    r13, r18 = k['S13'], 2 * k['S18']
    r17 = 2 * (k['K'] * k['S18'] ** k['A'])
    a = (r18 + r13 * r17) * (1. + r13 + r17)

    # the coefficients for the calculations
    eqn_mat = np.array([[r13, 0.5 * r17],
                        [r13 * ((r17 ** 2 + r17 - r18) / a),
                         1 - 0.5 * r17 * ((r13 ** 2 + r13 - r18) / a)]])

    # precalculate the d45 and d46 of the standard versus PDB
    r45d45std = (eqn_mat[0, 0] * d13cstd + eqn_mat[0, 1] * d18ostd)
    d46std = eqn_mat[1, 0] * d13cstd + eqn_mat[1, 1] * d18ostd

    # calculate the d45 and d46 of our sample versus PDB
    # in r45d45, r45 of PDB = r13 + r17 of PDB
    r45d45 = 1000. * (r45sam / r45std - 1.) * \
        (r13 + r17 + 0.001 * r45d45std) + r45d45std
    d46 = 1000. * (r46sam / r46std - 1.) * (1. + 0.001 * d46std) + d46std

    # solve the system of equations
    x = np.linalg.solve(eqn_mat, np.array([r45d45, d46]))
    return x[0]