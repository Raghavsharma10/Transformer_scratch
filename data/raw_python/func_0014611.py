def register_comet_object(self, *args, **kwargs):
        """Registers all functions from the object as Comet functions
        (see :ref:`comet-plugin`).

        This makes mass registration of functions a lot easier.

        Refer to :func:`sijax.plugin.comet.register_comet_object`
        for more details -ts signature differs slightly.

        This method's signature is the same, except that the first
        argument that :func:`sijax.plugin.comet.register_comet_object`
        expects is the Sijax instance, and this method
        does that automatically, so you don't have to do it.
        """
        sijax.plugin.comet.register_comet_object(self._sijax, *args, **kwargs)