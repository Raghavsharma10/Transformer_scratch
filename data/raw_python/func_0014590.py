def help(self, subject=None, args=None):
        """Get help information about Automation API.

        The following values can be specified for the subject:
            None -- gets an overview of help.
            'commands' -- gets a list of API functions
            command name -- get info about the specified command.
            object type  -- get info about the specified object type
            handle value -- get info about the object type referred to

        Arguments:
        subject -- Optional.  Subject to get help on.
        args    -- Optional.  Additional arguments for searching help.  These
                   are used when the subject is 'list'.

        Return:
        String of help information.

        """
        if subject:
            if subject not in (
                'commands', 'create', 'config', 'get', 'delete', 'perform',
                'connect', 'connectall', 'disconnect', 'disconnectall',
                'apply', 'log', 'help'):
                self._check_session()
            status, data = self._rest.get_request('help', subject, args)
        else:
            status, data = self._rest.get_request('help')

        if isinstance(data, (list, tuple, set)):
            return ' '.join((str(i) for i in data))
        return data['message']