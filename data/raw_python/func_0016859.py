def _initialise(self, feed_type="linear"):
        """
        Initialise the object by generating appropriate filenames,
        opening associated file handles and inspecting the FITS axes
        of these files.
        """
        self._filenames = filenames = _create_filenames(self._filename_schema,
                                                        feed_type)
        self._files = files = _open_fits_files(filenames)
        self._axes = axes = _create_axes(filenames, files)
        self._dim_indices = dim_indices = l_ax, m_ax, f_ax = tuple(
            axes.iaxis(d) for d in self._fits_dims)

        # Complain if we can't find required axes
        for i, ax in zip(dim_indices, self._fits_dims):
            if i == -1:
                raise ValueError("'%s' axis not found!" % ax)

        self._cube_extents = _cube_extents(axes, l_ax, m_ax, f_ax,
            self._l_sign, self._m_sign)
        self._shape = tuple(axes.naxis[d] for d in dim_indices) + (4,)
        self._beam_freq_map = axes.grid[f_ax]

        # Now describe our dimension sizes
        self._dim_updates = [(n, axes.naxis[i]) for n, i
            in zip(self._beam_dims, dim_indices)]

        self._initialised = True