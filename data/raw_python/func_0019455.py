def configure_truth(self, **kwargs):  # pragma: no cover
        """ Configure the arguments passed to the ``axvline`` and ``axhline``
        methods when plotting truth values.

        If you do not call this explicitly, the :func:`plot` method will
        invoke this method automatically.

        Recommended to set the parameters ``linestyle``, ``color`` and/or ``alpha``
        if you want some basic control.

        Default is to use an opaque black dashed line.

        Parameters
        ----------
        kwargs : dict
            The keyword arguments to unwrap when calling ``axvline`` and ``axhline``.

        Returns
        -------
        ChainConsumer
            Itself, to allow chaining calls.
        """
        if kwargs.get("ls") is None and kwargs.get("linestyle") is None:
            kwargs["ls"] = "--"
            kwargs["dashes"] = (3, 3)
        if kwargs.get("color") is None:
            kwargs["color"] = "#000000"
        self.config_truth = kwargs
        self._configured_truth = True
        return self