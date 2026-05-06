def twotheta(matrix, bcx, bcy, pixsizeperdist):
    """Calculate the two-theta matrix for a scattering matrix

    Inputs:
        matrix: only the shape of it is needed
        bcx, bcy: beam position (counting from 0; x is row, y is column index)
        pixsizeperdist: the pixel size divided by the sample-to-detector
            distance

    Outputs:
        the two theta matrix, same shape as 'matrix'.
    """
    col, row = np.meshgrid(list(range(matrix.shape[1])), list(range(matrix.shape[0])))
    return np.arctan(np.sqrt((row - bcx) ** 2 + (col - bcy) ** 2) * pixsizeperdist)