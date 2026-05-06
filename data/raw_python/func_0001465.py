def set_bg_data(self, bg_data, which_data=None):
        """Set background amplitude and phase data

        Parameters
        ----------
        bg_data: 2d ndarray (float or complex), list, QPImage, or `None`
            The background data (must be same type as `data`).
            If set to `None`, the background data is reset.
        which_data: str
            String or comma-separated list of strings indicating
            the order and type of input data. Valid values are
            "field", "phase", "phase,amplitude", or "phase,intensity",
            where the latter two require an indexable object for
            `bg_data` with the phase data as first element.
        """
        if isinstance(bg_data, QPImage):
            if which_data is not None:
                msg = "`which_data` must not be set if `bg_data` is QPImage!"
                raise ValueError(msg)
            pha, amp = bg_data.pha, bg_data.amp
        elif bg_data is None:
            # Reset phase and amplitude
            amp, pha = None, None
        else:
            # Compute phase and amplitude from data and which_data
            amp, pha = self._get_amp_pha(bg_data, which_data)
        # Set background data
        self._amp.set_bg(amp, key="data")
        self._pha.set_bg(pha, key="data")