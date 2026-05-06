def _get_amp_pha(self, data, which_data):
        """Convert input data to phase and amplitude

        Parameters
        ----------
        data: 2d ndarray (float or complex) or list
            The experimental data (see `which_data`)
        which_data: str
            String or comma-separated list of strings indicating
            the order and type of input data. Valid values are
            "field", "phase", "hologram", "phase,amplitude", or
            "phase,intensity", where the latter two require an
            indexable object with the phase data as first element.

        Returns
        -------
        amp, pha: tuple of (:class:`Amplitdue`, :class:`Phase`)
        """
        which_data = QPImage._conv_which_data(which_data)
        if which_data not in VALID_INPUT_DATA:
            msg = "`which_data` must be one of {}!".format(VALID_INPUT_DATA)
            raise ValueError(msg)

        if which_data == "field":
            amp = np.abs(data)
            pha = np.angle(data)
        elif which_data == "phase":
            pha = data
            amp = np.ones_like(data)
        elif which_data == ("phase", "amplitude"):
            amp = data[1]
            pha = data[0]
        elif which_data == ("phase", "intensity"):
            amp = np.sqrt(data[1])
            pha = data[0]
        elif which_data == "hologram":
            amp, pha = self._get_amp_pha(holo.get_field(data, **self.holo_kw),
                                         which_data="field")
        if amp.size == 0 or pha.size == 0:
            msg = "`data` with shape {} has zero size!".format(amp.shape)
            raise ValueError(msg)
        # phase unwrapping (take into account nans)
        nanmask = np.isnan(pha)
        if np.sum(nanmask):
            # create masked array
            # skimage.restoration.unwrap_phase cannot handle nan data
            # (even if masked)
            pham = pha.copy()
            pham[nanmask] = 0
            pham = np.ma.masked_array(pham, mask=nanmask)
            pha = unwrap_phase(pham, seed=47)
            pha[nanmask] = np.nan
        else:
            pha = unwrap_phase(pha, seed=47)

        return amp, pha