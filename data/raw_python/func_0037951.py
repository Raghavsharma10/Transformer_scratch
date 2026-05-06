def set_deserializer_by_mime_type(self, mime_type):
        """
        :param mime_type:
        :return:

        Used by content_type_set to set get a reference to the serializer object
        """

        for deserializer in self._deserializers:
            if deserializer.content_type() == mime_type:
                self._selected_deserializer = deserializer
                return

        raise exception.UnsupportedContentTypeError(mime_type, self.supported_mime_types_str)