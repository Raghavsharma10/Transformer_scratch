def jcrop_css(css_url=None):
        """Load jcrop css file.

        :param css_url: The custom CSS URL.
        """
        if css_url is None:
            if current_app.config['AVATARS_SERVE_LOCAL']:
                css_url = url_for('avatars.static', filename='jcrop/css/jquery.Jcrop.min.css')
            else:
                css_url = 'https://cdn.jsdelivr.net/npm/jcrop-0.9.12@0.9.12/css/jquery.Jcrop.min.css'
        return Markup('<link rel="stylesheet" href="%s">' % css_url)