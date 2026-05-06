def _run__exec(self, action, replace):
        """
        Run a system command

        >>> Action().run("hello", actions={
        ...     "hello": {
        ...         "type": "exec",
        ...         "cmd": "echo version=%{version}"
        ... }}, replace={
        ...     "version": "1712.10"
        ... })
        version=1712.10
        """

        cmd = action.get('cmd')
        shell = False
        if isinstance(cmd, str):
            shell = True

        if replace and action.get("template", True):
            if shell:
                cmd = self.rfxcfg.macro_expand(cmd, replace)
            else:
                cmd = [self.rfxcfg.macro_expand(x, replace) for x in cmd]

        self.logf("Action {} exec\n", action['name'])
        self.logf("{}\n", cmd, level=common.log_cmd)
        if self.sys(cmd):
            self.logf("Success\n", level=common.log_good)
            return
        self.die("Failure\n", level=common.log_err)