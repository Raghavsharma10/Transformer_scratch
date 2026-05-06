def set_bg(self, bg, key="data", attrs={}):
        """Set the background data

        Parameters
        ----------
        bg: numbers.Real, 2d ndarray, ImageData, or h5py.Dataset
            The background data. If `bg` is an `h5py.Dataset` object,
            it must exist in the same hdf5 file (a hard link is created).
            If set to `None`, the data will be removed.
        key: str
            One of :const:`VALID_BG_KEYS`)
        attrs: dict
            List of background attributes

        See Also
        --------
        del_bg: removing background data
        """
        if key not in VALID_BG_KEYS:
            raise ValueError("Invalid bg key: {}".format(key))
        # remove previous background key
        if key in self.h5["bg_data"]:
            del self.h5["bg_data"][key]
        # set background
        if isinstance(bg, (numbers.Real, np.ndarray)):
            dset = write_image_dataset(group=self.h5["bg_data"],
                                       key=key,
                                       data=bg,
                                       h5dtype=self.h5dtype)
            for kw in attrs:
                dset.attrs[kw] = attrs[kw]
        elif isinstance(bg, h5py.Dataset):
            # Create a hard link
            # (This functionality was intended for saving memory when storing
            # large QPSeries with universal background data, i.e. when using
            # `QPSeries.add_qpimage` with the `bg_from_idx` keyword.)
            self.h5["bg_data"][key] = bg
        elif bg is not None:
            msg = "Unknown background data type: {}".format(bg)
            raise ValueError(msg)