def send(self, id_, text, identities, context=None):
        """
        Send messages when using RapidSMS 0.14.0 or later.

        We can send multiple messages in one Tropo program, so we do
        that.

        :param id_: Unused, included for compatibility with RapidSMS.
        :param string text: The message text to send.
        :param identities: A list of identities to send the message to
            (a list of strings)
        :param context: Unused, included for compatibility with RapidSMS.
        """

        # Build our program
        from_ = self.config['number'].replace('-', '')
        commands = []
        for identity in identities:
            # We'll include a 'message' command for each recipient.
            # The Tropo doc explicitly says that while passing a list
            # of destination numbers is not a syntax error, only the
            # first number on the list will get sent the message. So
            # we have to send each one as a separate `message` command.
            commands.append(
                {
                    'message': {
                        'say': {'value': text},
                        'to': identity,
                        'from': from_,
                        'channel': 'TEXT',
                        'network': 'SMS'
                    }
                }
            )
            program = {
                'tropo': commands,
            }
        self.execute_tropo_program(program)