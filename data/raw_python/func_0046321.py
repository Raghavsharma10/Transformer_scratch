def expand(self, expand):
        '''
        Turn off argument expansion, useful for 'grep'. Example:

            Sh('grep .*').expand(False) > 'tango'

        :param expand: True or False
        :return: self
        '''
        self._expand = expand
        cmd = self._original_cmd
        if len(cmd) == 1:
            if not type(cmd[0]) in (str, unicode):
                cmd = cmd[0]
            else:
                cmd = shlex.split(cmd[0])
        cmd = [os.path.expanduser(os.path.expandvars(arg)) for arg in cmd]
        self._cmd, self._args = cmd[0], cmd[1:]
        return self