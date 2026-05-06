def jcrop_js(js_url=None, with_jquery=True):
        """Load jcrop Javascript file.

        :param js_url: The custom JavaScript URL.
        :param with_jquery: Include jQuery or not, default to ``True``.
        """
        serve_local = current_app.config['AVATARS_SERVE_LOCAL']

        if js_url is None:
            if serve_local:
                js_url = url_for('avatars.static', filename='jcrop/js/jquery.Jcrop.min.js')
            else:
                js_url = 'https://cdn.jsdelivr.net/npm/jcrop-0.9.12@0.9.12/js/jquery.Jcrop.min.js'

        if with_jquery:
            if serve_local:
                jquery = '<script src="%s"></script>' % url_for('avatars.static', filename='jcrop/js/jquery.min.js')
            else:
                jquery = '<script src="https://cdn.jsdelivr.net/npm/jcrop-0.9.12@0.9.12/js/jquery.min.js"></script>'
        else:
            jquery = ''
        return Markup('''%s\n<script src="%s"></script>
        ''' % (jquery, js_url))