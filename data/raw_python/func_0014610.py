def register_comet_callback(self, *args, **kwargs):
        """Registers a single Comet callback function
        (see :ref:`comet-plugin`).

        Refer to :func:`sijax.plugin.comet.register_comet_callback`
        for more details - its signature differs slightly.

        This method's signature is the same, except that the first
        argument that :func:`sijax.plugin.comet.register_comet_callback`
        expects is the Sijax instance, and this method
        does that automatically, so you don't have to do it.
        """
        sijax.plugin.comet.register_comet_callback(self._sijax, *args, **kwargs)