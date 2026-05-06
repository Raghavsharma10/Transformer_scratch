def image_name(self):
        """
        The image_name of a container is the concatenation of the ``image_index``,
        ``image_name_prefix``, and ``name`` of the image.

        Also, if $EXTRA_IMAGE_NAME is defined, that is appended
        """
        if getattr(self, "_image_name", NotSpecified) is NotSpecified:
            self._image_name = self.prefixed_image_name

            if self.image_index:
                self._image_name = "{0}{1}".format(self.image_index, self._image_name)

            if "EXTRA_IMAGE_NAME" in os.environ:
                self._image_name = "{0}{1}".format(self._image_name, os.environ["EXTRA_IMAGE_NAME"])
        return self._image_name