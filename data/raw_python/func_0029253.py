def input(self, opt):
        """
        Description:

            Set the input
            Call with no arguments to get current setting

        Arguments:
            opt: string
                Name provided from input list or key from yaml ("HDMI 1" or "hdmi_1")
        """

        for key in self.command['input']:
            if (key == opt) or (self.command['input'][key]['name'] == opt):
                return self._send_command(['input', key, 'command'])
        return False