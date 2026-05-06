def as_matrix_transform(transform):
    """
    Simplify a transform to a single matrix transform, which makes it a lot
    faster to compute transformations.

    Raises a TypeError if the transform cannot be simplified.
    """
    if isinstance(transform, ChainTransform):
        matrix = np.identity(4)
        for tr in transform.transforms:
            # We need to do the matrix multiplication manually because VisPy
            # somehow doesn't mutliply matrices if there is a perspective
            # component. The equation below looks like it's the wrong way
            # around, but the VisPy matrices are transposed.
            matrix = np.matmul(as_matrix_transform(tr).matrix, matrix)
        return MatrixTransform(matrix)
    elif isinstance(transform, InverseTransform):
        matrix = as_matrix_transform(transform._inverse)
        return MatrixTransform(matrix.inv_matrix)
    elif isinstance(transform, NullTransform):
        return MatrixTransform()
    elif isinstance(transform, STTransform):
        return transform.as_matrix()
    elif isinstance(transform, MatrixTransform):
        return transform
    else:
        raise TypeError("Could not simplify transform of type {0}".format(type(transform)))