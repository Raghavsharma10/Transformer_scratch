def offset_mode(data):
    """Compute Mode using a histogram with `sqrt(data.size)` bins"""
    nbins = int(np.ceil(np.sqrt(data.size)))
    mind, maxd = data.min(), data.max()
    histo = np.histogram(data, nbins, density=True, range=(mind, maxd))
    dx = abs(histo[1][1] - histo[1][2]) / 2
    hx = histo[1][1:] - dx
    hy = histo[0]
    idmax = np.argmax(hy)
    return hx[idmax]