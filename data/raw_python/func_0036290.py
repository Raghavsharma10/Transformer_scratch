def sbesselh2(x, N):
    "Spherical Hankel of the second kind"

    jn = sbesselj(x, N)
    yn = sbessely(x, N)

    return jn - 1j * yn