def _thumbnail_local(self, original_filename, thumb_filename,
                         thumb_size, thumb_url, crop=None, bg=None,
                         quality=85):
        """Finds or creates a thumbnail for the specified image on the local filesystem."""

        # create folders
        self._get_path(thumb_filename)

        thumb_url_full = url_for('static', filename=thumb_url)

        # Return the thumbnail URL now if it already exists locally
        if os.path.exists(thumb_filename):
            return thumb_url_full

        try:
            image = Image.open(original_filename)
        except IOError:
            return ''

        img = self._thumbnail_resize(image, thumb_size, crop=crop, bg=bg)

        img.save(thumb_filename, image.format, quality=quality)

        return thumb_url_full