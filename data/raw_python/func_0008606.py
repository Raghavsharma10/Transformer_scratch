def convert(self, format):
        """Convert the project in-place to a different file format.

        Returns a list of :class:`UnsupportedFeature` objects, which may give
        warnings about the conversion.

        :param format: :attr:`KurtFileFormat.name` eg. ``"scratch14"``.

        :raises: :class:`ValueError` if the format doesn't exist.

        """
        self._plugin = kurt.plugin.Kurt.get_plugin(format)
        return list(self._normalize())