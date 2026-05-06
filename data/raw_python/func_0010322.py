def set_quality(self, quality):
        """Set the quality for this sample

        Quality is stored on Device Cloud as a 32-bit integer, so the input
        to this function should be either None, an integer, or a string that can
        be converted to an integer.

        """
        if isinstance(quality, *six.string_types):
            quality = int(quality)
        elif isinstance(quality, float):
            quality = int(quality)

        self._quality = validate_type(quality, type(None), *six.integer_types)