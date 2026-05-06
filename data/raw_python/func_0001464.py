def refocus(self, distance, method="helmholtz", h5file=None, h5mode="a"):
        """Compute a numerically refocused QPImage

        Parameters
        ----------
        distance: float
            Focusing distance [m]
        method: str
            Refocusing method, one of ["helmholtz","fresnel"]
        h5file: str, h5py.Group, h5py.File, or None
            A path to an hdf5 data file where the QPImage is cached.
            If set to `None` (default), all data will be handled in
            memory using the "core" driver of the :mod:`h5py`'s
            :class:`h5py:File` class. If the file does not exist,
            it is created. If the file already exists, it is opened
            with the file mode defined by `hdf5_mode`. If this is
            an instance of h5py.Group or h5py.File, then this will
            be used to internally store all data.
        h5mode: str
            Valid file modes are (only applies if `h5file` is a path)

            - "r": Readonly, file must exist
            - "r+": Read/write, file must exist
            - "w": Create file, truncate if exists
            - "w-" or "x": Create file, fail if exists
            - "a": Read/write if exists, create otherwise (default)

        Returns
        -------
        qpi: qpimage.QPImage
            Refocused phase and amplitude data

        See Also
        --------
        :mod:`nrefocus`: library used for numerical focusing
        """
        field2 = nrefocus.refocus(field=self.field,
                                  d=distance/self["pixel size"],
                                  nm=self["medium index"],
                                  res=self["wavelength"]/self["pixel size"],
                                  method=method
                                  )
        if "identifier" in self:
            ident = self["identifier"]
        else:
            ident = ""
        meta_data = self.meta
        meta_data["identifier"] = "{}@{}{:.5e}m".format(ident,
                                                        method[0],
                                                        distance)
        qpi2 = QPImage(data=field2,
                       which_data="field",
                       meta_data=meta_data,
                       h5file=h5file,
                       h5mode=h5mode)
        return qpi2