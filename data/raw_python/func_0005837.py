def set_hook(self, phase, action):
        """Allows setting hooks (attaching actions) for various uWSGI phases.

        :param str|unicode phase: See constants in ``.phases``.

        :param str|unicode|list|HookAction|list[HookAction] action:

        """
        self._set('hook-%s' % phase, action, multi=True)

        return self._section