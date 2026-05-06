def _set_default_serializer(self, name):
        """
        Set the default serialization method used by this library.

        :param name: The name of the registered serialization method.
            For example, ``json`` (default), ``pickle``, ``yaml``,
            or any custom methods registered using :meth:`register`.

        :raises SerializerNotInstalled: If the serialization method
            requested is not available.
        """
        try:
            (self._default_content_type, self._default_content_encoding,
             self._default_encode) = self._encoders[name]
        except KeyError:
            raise SerializerNotInstalled(
                "No encoder installed for %s" % name)