def run_command_on_event(self, command, phase=phases.ASAP):
        """Run the given command on a given phase.

        :param str|unicode command:

        :param str|unicode phase: See constants in ``Phases`` class.

        """
        self._set('exec-%s' % phase, command, multi=True)

        return self._section