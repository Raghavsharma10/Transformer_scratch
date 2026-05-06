def force_extrapolation(self):
        """Force the underlying model to extrapolate.

        An example where this is useful: You create a source spectrum
        with non-default extrapolation behavior and you wish to force
        the underlying empirical model to extrapolate based on nearest point.

        .. note::

            This is only applicable to `~synphot.models.Empirical1D` model
            and should still work even if the source spectrum has been
            redshifted.

        Returns
        -------
        is_forced : bool
            `True` if the model is successfully forced to be extrapolated,
            else `False`.

        """
        # We use _model here in case the spectrum is redshifted.
        if isinstance(self._model, Empirical1D):
            self._model.fill_value = np.nan
            is_forced = True
        else:
            is_forced = False

        return is_forced