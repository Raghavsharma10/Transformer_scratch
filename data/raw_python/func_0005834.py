def change_dir(self, to, after_app_loading=False):
        """Chdir to specified directory before or after apps loading.

        :param str|unicode to: Target directory.

        :param bool after_app_loading:
                *True* - after load
                *False* - before load

        """
        self._set('chdir2' if after_app_loading else 'chdir', to)

        return self._section