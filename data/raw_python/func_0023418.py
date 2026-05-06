def read_mesh(fname):
    """Read mesh data from file.

    Parameters
    ----------
    fname : str
        File name to read. Format will be inferred from the filename.
        Currently only '.obj' and '.obj.gz' are supported.

    Returns
    -------
    vertices : array
        Vertices.
    faces : array | None
        Triangle face definitions.
    normals : array
        Normals for the mesh.
    texcoords : array | None
        Texture coordinates.
    """
    # Check format
    fmt = op.splitext(fname)[1].lower()
    if fmt == '.gz':
        fmt = op.splitext(op.splitext(fname)[0])[1].lower()

    if fmt in ('.obj'):
        return WavefrontReader.read(fname)
    elif not format:
        raise ValueError('read_mesh needs could not determine format.')
    else:
        raise ValueError('read_mesh does not understand format %s.' % fmt)