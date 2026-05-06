def run_command_on_touch(self, command, target):
        """Run command when the specified file is modified/touched.

        :param str|unicode command:

        :param str|unicode target: File path.

        """
        self._set('touch-exec', '%s %s' % (target, command), multi=True)

        return self._section