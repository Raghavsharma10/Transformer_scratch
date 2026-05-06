def cudaMemcpy_htod(dst, src, count):
    """
    Copy memory from host to device.

    Copy data from host memory to device memory.

    Parameters
    ----------
    dst : ctypes pointer
        Device memory pointer.
    src : ctypes pointer
        Host memory pointer.
    count : int
        Number of bytes to copy.

    """

    status = _libcudart.cudaMemcpy(dst, src,
                                   ctypes.c_size_t(count),
                                   cudaMemcpyHostToDevice)
    cudaCheckStatus(status)