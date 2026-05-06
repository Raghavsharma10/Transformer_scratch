def linear(arr1, arr2):
    """
    Create a linear blend of arr1 (fading out) and arr2 (fading in)
    """
    n = N.shape(arr1)[0]
    try:
        channels = N.shape(arr1)[1]
    except:
        channels = 1

    f_in = N.linspace(0, 1, num=n)
    f_out = N.linspace(1, 0, num=n)

    # f_in = N.arange(n) / float(n - 1)
    # f_out = N.arange(n - 1, -1, -1) / float(n)

    if channels > 1:
        f_in = N.tile(f_in, (channels, 1)).T
        f_out = N.tile(f_out, (channels, 1)).T

    vals = f_out * arr1 + f_in * arr2
    return vals