def sbesselh1(x, N):
    "Spherical Hankel of the first kind"
    
    jn = sbesselj(x, N)
    yn = sbessely(x, N)

    return jn + 1j * yn