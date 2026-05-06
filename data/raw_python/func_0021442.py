def convolve_fft(array, kernel):
    """
    Convolve an array with a kernel using FFT.
    Implemntation based on the convolve_fft function from astropy.

    https://github.com/astropy/astropy/blob/master/astropy/convolution/convolve.py
    """

    array = np.asarray(array, dtype=np.complex)
    kernel = np.asarray(kernel, dtype=np.complex)

    if array.ndim != kernel.ndim:
        raise ValueError("Image and kernel must have same number of "
                         "dimensions")

    array_shape = array.shape
    kernel_shape = kernel.shape
    new_shape = np.array(array_shape) + np.array(kernel_shape)

    array_slices = []
    kernel_slices = []
    for (new_dimsize, array_dimsize, kernel_dimsize) in zip(
            new_shape, array_shape, kernel_shape):
        center = new_dimsize - (new_dimsize + 1) // 2
        array_slices += [slice(center - array_dimsize // 2,
                         center + (array_dimsize + 1) // 2)]
        kernel_slices += [slice(center - kernel_dimsize // 2,
                          center + (kernel_dimsize + 1) // 2)]

    array_slices = tuple(array_slices)
    kernel_slices = tuple(kernel_slices)

    if not np.all(new_shape == array_shape):
        big_array = np.zeros(new_shape, dtype=np.complex)
        big_array[array_slices] = array
    else:
        big_array = array

    if not np.all(new_shape == kernel_shape):
        big_kernel = np.zeros(new_shape, dtype=np.complex)
        big_kernel[kernel_slices] = kernel
    else:
        big_kernel = kernel

    array_fft = np.fft.fftn(big_array)
    kernel_fft = np.fft.fftn(np.fft.ifftshift(big_kernel))

    rifft = np.fft.ifftn(array_fft * kernel_fft)

    return rifft[array_slices].real