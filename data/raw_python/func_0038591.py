def _set_serializer_by_mime_type(self, mime_type):
        """
        :param mime_type:
        :return:

        used by content_type_set to set get a reference to the appropriate serializer
        """

        # ignore if binary response
        if isinstance(self._app_iter, BinaryResponse):
            self.logger.info("ignoring setting serializer for binary response")
            return

        for available_serializer in self._serializers:
            if available_serializer.content_type() == mime_type:
                self._selected_serializer = available_serializer
                self.logger.info("set serializer for mime type: %s" % mime_type)
                return

        self.logger.info("could not find serializer for mime type: %s" % mime_type)
        raise exception.UnsupportedVocabularyError(mime_type, self.supported_mime_types_str)