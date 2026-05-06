def crop_box(endpoint=None, filename=None):
        """Create a crop box.

        :param endpoint: The endpoint of view function that serve avatar image file.
        :param filename: The filename of the image that need to be crop.
        """
        crop_size = current_app.config['AVATARS_CROP_BASE_WIDTH']

        if endpoint is None or filename is None:
            url = url_for('avatars.static', filename='default/default_l.jpg')
        else:
            url = url_for(endpoint, filename=filename)
        return Markup('<img src="%s" id="crop-box" style="max-width: %dpx; display: block;">' % (url, crop_size))