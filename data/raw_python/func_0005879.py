def mount(self, mountpoint, app, into_worker=False):
        """Load application under mountpoint.

        Example:
            * .mount('', 'app0.py') -- Root URL part
            * .mount('/app1', 'app1.py') -- URL part
            * .mount('/pinax/here', '/var/www/pinax/deploy/pinax.wsgi')
            * .mount('the_app3', 'app3.py')  -- Variable value: application alias (can be set by ``UWSGI_APPID``)
            * .mount('example.com', 'app2.py')  -- Variable value: Hostname (variable set in nginx)

        * http://uwsgi-docs.readthedocs.io/en/latest/Nginx.html#hosting-multiple-apps-in-the-same-process-aka-managing-script-name-and-path-info

        :param str|unicode mountpoint: URL part, or variable value.

            .. note:: In case of URL part you may also want to set ``manage_script_name`` basic param to ``True``.

            .. warning:: In case of URL part a trailing slash may case problems in some cases
                (e.g. with Django based projects).

        :param str|unicode app: App module/file.

        :param bool into_worker: Load application under mountpoint
            in the specified worker or after workers spawn.

        """
        # todo check worker mount -- uwsgi_init_worker_mount_app() expects worker://
        self._set('worker-mount' if into_worker else 'mount', '%s=%s' % (mountpoint, app), multi=True)

        return self._section