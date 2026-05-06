def change_autocommit_mode(self, switch):
        """
        Strip and make a string case insensitive and ensure it is either 'true' or 'false'.

        If neither, prompt user for either value.
        When 'true', return True, and when 'false' return False.
        """
        parsed_switch = switch.strip().lower()
        if not parsed_switch in ['true', 'false']:
            self.send_response(
                self.iopub_socket, 'stream', {
                    'name': 'stderr',
                    'text': 'autocommit must be true or false.\n\n'
                }
            )

        switch_bool = (parsed_switch == 'true')
        committed = self.switch_autocommit(switch_bool)
        message = (
            'committed current transaction & ' if committed else '' +
            'switched autocommit mode to ' +
            str(self._autocommit)
        )
        self.send_response(
            self.iopub_socket, 'stream', {
                'name': 'stderr',
                'text': message,
            }
        )