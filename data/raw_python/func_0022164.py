def preview_box(endpoint=None, filename=None):
        """Create a preview box.

        :param endpoint: The endpoint of view function that serve avatar image file.
        :param filename: The filename of the image that need to be crop.
        """
        preview_size = current_app.config['AVATARS_CROP_PREVIEW_SIZE'] or current_app.config['AVATARS_SIZE_TUPLE'][2]

        if endpoint is None or filename is None:
            url = url_for('avatars.static', filename='default/default_l.jpg')
        else:
            url = url_for(endpoint, filename=filename)
        return Markup('''
        <div id="preview-box">
        <div class="preview-box" style="width: %dpx; height: %dpx; overflow: hidden;">
          <img src="%s" class="jcrop-preview" alt="Preview"/>
        </div>
      </div>''' % (preview_size, preview_size, url))