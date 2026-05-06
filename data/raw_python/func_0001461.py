def clear_bg(self, which_data=("amplitude", "phase"), keys="fit"):
        """Clear background correction

        Parameters
        ----------
        which_data: str or list of str
            From which type of data to remove the background
            information. The list contains either "amplitude",
            "phase", or both.
        keys: str or list of str
            Which type of background data to remove. One of:

            - "fit": the background data computed with
              :func:`qpimage.QPImage.compute_bg`
            - "data": the experimentally obtained background image
        """
        which_data = QPImage._conv_which_data(which_data)
        if isinstance(keys, str):
            # make sure keys is a list of strings
            keys = [keys]

        # Get image data for clearing
        imdats = []
        if "amplitude" in which_data:
            imdats.append(self._amp)
        if "phase" in which_data:
            imdats.append(self._pha)
        if not imdats:
            msg = "`which_data` must contain 'phase' or 'amplitude'!"
            raise ValueError(msg)
        # Perform clearing of backgrounds
        for imdat in imdats:
            for key in keys:
                imdat.del_bg(key)