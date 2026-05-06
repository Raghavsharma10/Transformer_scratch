def thumbnail(self):
        """Read-only attribute that provides the value of the thumbnail to display.
        """
        # check if there is a valid thumbnail override
        if self.thumbnail_override.id is not None:
            return self.thumbnail_override

        # otherwise, just try to grab the first image
        first_image = self.first_image
        if first_image is not None:
            return first_image

        # no override for thumbnail and no non-none image field, just return override,
        # which is a blank image field
        return self.thumbnail_override