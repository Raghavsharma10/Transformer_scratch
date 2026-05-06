def get_remote_button_list(self):
        """
        Description:

            Get remote button list
            Returns an list of all available remote buttons

        """
        remote_buttons = []
        for key in self.command['remote']:
            if self.command['remote'][key] != '':
                remote_buttons.append(key)
        return remote_buttons