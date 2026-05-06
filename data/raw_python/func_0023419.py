def write_mesh(fname, vertices, faces, normals, texcoords, name='',
               format='obj', overwrite=False, reshape_faces=True):
    """ Write mesh data to file.

    Parameters
    ----------
    fname : str
        Filename to write. Must end with ".obj" or ".gz".
    vertices : array
        Vertices.
    faces : array | None
        Triangle face definitions.
    normals : array
        Normals for the mesh.
    texcoords : array | None
        Texture coordinates.
    name : str
        Name of the object.
    format : str
        Currently only "obj" is supported.
    overwrite : bool
        If the file exists, overwrite it.
    reshape_faces : bool
        Reshape the `faces` array to (Nf, 3). Set to `False`
        if you need to write a mesh with non triangular faces.
    """
    # Check file
    if op.isfile(fname) and not overwrite:
        raise IOError('file "%s" exists, use overwrite=True' % fname)

    # Check format
    if format not in ('obj'):
        raise ValueError('Only "obj" format writing currently supported')
    WavefrontWriter.write(fname, vertices, faces,
                          normals, texcoords, name, reshape_faces)