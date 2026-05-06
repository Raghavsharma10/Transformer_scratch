def _pad_arrays(t, arrays, indices, span, period):
    """Internal routine to pad arrays for periodic models."""
    N = len(t)

    if indices is None:
        indices = np.arange(N)
    pad_left = max(0, 0 - np.min(indices - span // 2))
    pad_right = max(0, np.max(indices + span - span // 2) - (N - 1))

    if pad_left + pad_right > 0:
        Nright, pad_right = divmod(pad_right, N)
        Nleft, pad_left = divmod(pad_left, N)
        t = np.concatenate([t[N - pad_left:] - (Nleft + 1) * period]
                           + [t + i * period
                              for i in range(-Nleft, Nright + 1)]
                           + [t[:pad_right] + (Nright + 1) * period])
        arrays = [np.concatenate([a[N - pad_left:]]
                                 + (Nleft + Nright + 1) * [a]
                                 + [a[:pad_right]])
                  for a in arrays]
        pad_left = pad_left % N
        Nright = pad_right / N
        pad_right = pad_right % N

        return (t, arrays, slice(pad_left + Nleft * N,
                                 pad_left + (Nleft + 1) * N))
    else:
        return (t, arrays, slice(None))