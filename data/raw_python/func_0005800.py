def register_static_map(self, mountpoint, target, retain_resource_path=False, safe_target=False):
        """Allows mapping mountpoint to a static directory (or file).

        * http://uwsgi.readthedocs.io/en/latest/StaticFiles.html#mode-3-using-static-file-mount-points

        :param str|unicode mountpoint:

        :param str|unicode target:

        :param bool retain_resource_path: Append the requested resource to the docroot.

            Example: if ``/images`` maps to ``/var/www/img`` requested ``/images/logo.png`` will be served from:

            * ``True``: ``/var/www/img/images/logo.png``

            * ``False``: ``/var/www/img/logo.png``

        :param bool safe_target: Skip security checks if the file is under the specified path.

            Whether to consider resolved (real) target a safe one to serve from.

            * http://uwsgi.readthedocs.io/en/latest/StaticFiles.html#security

        """
        command = 'static-map'

        if retain_resource_path:

            command += '2'

        self._set(command, '%s=%s' % (mountpoint, target), multi=True)

        if safe_target:
            self._set('static-safe', target, multi=True)

        return self._section