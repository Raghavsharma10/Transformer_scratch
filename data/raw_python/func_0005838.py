def set_hook_touch(self, fpath, action):
        """Allows running certain action when the specified file is touched.

        :param str|unicode fpath: File path.

        :param str|unicode|list|HookAction|list[HookAction] action:

        """
        self._set('hook-touch', '%s %s' % (fpath, action), multi=True)

        return self._section