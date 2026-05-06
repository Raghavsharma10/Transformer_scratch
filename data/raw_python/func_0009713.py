def write(self, filename, binary=True):
        """Writes a surface mesh to disk.

        Written file may be an ASCII or binary ply, stl, or vtk mesh
        file.

        Parameters
        ----------
        filename : str
            Filename of mesh to be written.  Filetype is inferred from
            the extension of the filename unless overridden with
            ftype.  Can be one of the following types (.ply, .stl,
            .vtk)

        ftype : str, optional
            Filetype.  Inferred from filename unless specified with a
            three character string.  Can be one of the following:
            'ply', 'stl', or 'vtk'.

        Notes
        -----
        Binary files write much faster than ASCII.
        """
        self.mesh.write(filename, binary)