def set_placeholder(self, key, value):
        """Placeholders are custom magic variables defined during configuration
        time.

        .. note:: These are accessible, like any uWSGI option, in your application code via
            ``.runtime.environ.uwsgi_env.config``.

        :param str|unicode key:

        :param str|unicode value:

        """
        self._set('set-placeholder', '%s=%s' % (key, value), multi=True)

        return self