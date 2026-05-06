def fourier2dpad(data, zero_pad=True):
    """Compute the 2D Fourier transform with zero padding

    Parameters
    ----------
    data: 2d fload ndarray
        real-valued image data
    zero_pad: bool
        perform zero-padding to next order of 2
    """
    if zero_pad:
        # zero padding size is next order of 2
        (N, M) = data.shape
        order = np.int(max(64., 2**np.ceil(np.log(2 * max(N, M)) / np.log(2))))

        # this is faster than np.pad
        datapad = np.zeros((order, order), dtype=float)
        datapad[:data.shape[0], :data.shape[1]] = data
    else:
        datapad = data

    # Fourier transform
    fft = np.fft.fftshift(np.fft.fft2(datapad))
    return fft