def tree_alp(
    x, n, standardization, phi=None, with_condon_shortley_phase=True, symbolic=False
):
    """Evaluates the entire tree of associated Legendre polynomials up to depth
    n.
    There are many recurrence relations that can be used to construct the
    associated Legendre polynomials. However, only few are numerically stable.
    Many implementations (including this one) use the classical Legendre
    recurrence relation with increasing L.

    Useful references are

    Taweetham Limpanuparb, Josh Milthorpe,
    Associated Legendre Polynomials and Spherical Harmonics Computation for
    Chemistry Applications,
    Proceedings of The 40th Congress on Science and Technology of Thailand;
    2014 Dec 2-4, Khon Kaen, Thailand. P. 233-241.
    <https://arxiv.org/abs/1410.1748>

    and

    Schneider et al.,
    A new Fortran 90 program to compute regular and irregular associated
    Legendre functions,
    Computer Physics Communications,
    Volume 181, Issue 12, December 2010, Pages 2091-2097,
    <https://doi.org/10.1016/j.cpc.2010.08.038>.

    The return value is a list of arrays, where `out[k]` hosts the `2*k+1`
    values of the `k`th level of the tree

                              (0, 0)
                    (-1, 1)   (0, 1)   (1, 1)
          (-2, 2)   (-1, 2)   (0, 2)   (1, 2)   (2, 2)
            ...       ...       ...     ...       ...
    """
    # assert numpy.all(numpy.abs(x) <= 1.0)

    d = {
        "natural": (_Natural, [x, symbolic]),
        "spherical": (_Spherical, [x, symbolic]),
        "complex spherical": (_ComplexSpherical, [x, phi, symbolic, False]),
        "complex spherical 1": (_ComplexSpherical, [x, phi, symbolic, True]),
        "normal": (_Normal, [x, symbolic]),
        "schmidt": (_Schmidt, [x, phi, symbolic]),
    }
    fun, args = d[standardization]
    c = fun(*args)

    if with_condon_shortley_phase:

        def z1_factor_CSP(L):
            return -1 * c.z1_factor(L)

    else:
        z1_factor_CSP = c.z1_factor

    # Here comes the actual loop.
    e = numpy.ones_like(x, dtype=int)
    out = [[e * c.p0]]
    for L in range(1, n + 1):
        out.append(
            numpy.concatenate(
                [
                    [out[L - 1][0] * c.z0_factor(L)],
                    out[L - 1] * numpy.multiply.outer(c.C0(L), x),
                    [out[L - 1][-1] * z1_factor_CSP(L)],
                ]
            )
        )

        if L > 1:
            out[-1][2:-2] -= numpy.multiply.outer(c.C1(L), e) * out[L - 2]

    return out