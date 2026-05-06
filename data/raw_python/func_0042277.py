def _thumbnail_resize(self, image, thumb_size, crop=None, bg=None):
        """Performs the actual image cropping operation with PIL."""

        if crop == 'fit':
            img = ImageOps.fit(image, thumb_size, Image.ANTIALIAS)
        else:
            img = image.copy()
            img.thumbnail(thumb_size, Image.ANTIALIAS)

        if bg:
            img = self._bg_square(img, bg)

        return img