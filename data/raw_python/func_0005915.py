def set_autoreload_params(self, scan_interval=None, ignore_modules=None):
        """Sets autoreload related parameters.

        :param int scan_interval: Seconds. Monitor Python modules' modification times to trigger reload.

            .. warning:: Use only in development.

        :param list|st|unicode ignore_modules: Ignore the specified module during auto-reload scan.

        """
        self._set('py-auto-reload', scan_interval)
        self._set('py-auto-reload-ignore', ignore_modules, multi=True)

        return self._section