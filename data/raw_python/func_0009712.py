def repair(self, verbose=False, joincomp=False,
               remove_smallest_components=True):
        """Performs mesh repair using MeshFix's default repair
        process.

        Parameters
        ----------
        verbose : bool, optional
            Enables or disables debug printing.  Disabled by default.

        joincomp : bool, optional
            Attempts to join nearby open components.

        remove_smallest_components : bool, optional
            Remove all but the largest isolated component from the
            mesh before beginning the repair process.  Default True

        Notes
        -----
        Vertex and face arrays are updated inplace.  Access them with:
        meshfix.v
        meshfix.f
        """
        assert self.f.shape[1] == 3, 'Face array must contain three columns'
        assert self.f.ndim == 2, 'Face array must be 2D'
        self.v, self.f = _meshfix.clean_from_arrays(self.v, self.f,
                                                    verbose, joincomp,
                                                    remove_smallest_components)