def write(cls, fname, vertices, faces, normals,
              texcoords, name='', reshape_faces=True):
        """ This classmethod is the entry point for writing mesh data to OBJ.

        Parameters
        ----------
        fname : string
            The filename to write to. Must end with ".obj" or ".gz".
        vertices : numpy array
            The vertex data
        faces : numpy array
            The face data
        texcoords : numpy array
            The texture coordinate per vertex
        name : str
            The name of the object (e.g. 'teapot')
        reshape_faces : bool
            Reshape the `faces` array to (Nf, 3). Set to `False`
            if you need to write a mesh with non triangular faces.
        """
        # Open file
        fmt = op.splitext(fname)[1].lower()
        if fmt not in ('.obj', '.gz'):
            raise ValueError('Filename must end with .obj or .gz, not "%s"'
                             % (fmt,))
        opener = open if fmt == '.obj' else gzip_open
        f = opener(fname, 'wb')
        try:
            writer = WavefrontWriter(f)
            writer.writeMesh(vertices, faces, normals,
                             texcoords, name, reshape_faces=reshape_faces)
        except EOFError:
            pass
        finally:
            f.close()