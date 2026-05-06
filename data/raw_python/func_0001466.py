def add_qpimage(self, qpi, identifier=None, bg_from_idx=None):
        """Add a QPImage instance to the QPSeries

        Parameters
        ----------
        qpi: qpimage.QPImage
            The QPImage that is added to the series
        identifier: str
            Identifier key for `qpi`
        bg_from_idx: int or None
            Use the background data from the data stored in this index,
            creating hard links within the hdf5 file.
            (Saves memory if e.g. all qpimages is corrected with the same data)
        """
        if not isinstance(qpi, QPImage):
            raise ValueError("`fli` must be instance of QPImage!")
        if "identifier" in qpi and identifier is None:
            identifier = qpi["identifier"]
        if identifier and identifier in self:
            msg = "The identifier '{}' already ".format(identifier) \
                  + "exists! You can either change the identifier of " \
                  + " '{}' or remove it.".format(qpi)
            raise ValueError(msg)
        # determine number of qpimages
        num = len(self)
        # indices start at zero; do not add 1
        name = "qpi_{}".format(num)
        group = self.h5.create_group(name)
        thisqpi = qpi.copy(h5file=group)

        if bg_from_idx is not None:
            # Create hard links
            refqpi = self[bg_from_idx]
            thisqpi._amp.set_bg(bg=refqpi._amp.h5["bg_data"]["data"])
            thisqpi._pha.set_bg(bg=refqpi._pha.h5["bg_data"]["data"])

        if identifier:
            # set identifier
            group.attrs["identifier"] = identifier