def get_bg(self, key=None, ret_attrs=False):
        """Get the background data

        Parameters
        ----------
        key: None or str
            A user-defined key that identifies the background data.
            Examples are "data" for experimental data, or "fit"
            for an estimated background correction
            (see :const:`VALID_BG_KEYS`). If set to `None`,
            returns the combined background image (:const:`ImageData.bg`).
        ret_attrs: bool
            Also returns the attributes of the background data.
        """
        if key is None:
            if ret_attrs:
                raise ValueError("No attributes for combined background!")
            return self.bg
        else:
            if key not in VALID_BG_KEYS:
                raise ValueError("Invalid bg key: {}".format(key))
            if key in self.h5["bg_data"]:
                data = self.h5["bg_data"][key][:]
                if ret_attrs:
                    attrs = dict(self.h5["bg_data"][key].attrs)
                    # remove keys for image visualization in hdf5 files
                    for h5k in ["CLASS", "IMAGE_VERSION", "IMAGE_SUBCLASS"]:
                        if h5k in attrs:
                            attrs.pop(h5k)
                    ret = (data, attrs)
                else:
                    ret = data
            else:
                raise KeyError("No background data for {}!".format(key))
            return ret