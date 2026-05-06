def perform(self, command, params=None, **kwargs):
        """Execute a command.

        Arguments can be supplied either as a dictionary or as keyword
        arguments.  Examples:
            stc.perform('LoadFromXml', {'filename':'config.xml'})
            stc.perform('LoadFromXml', filename='config.xml')

        Arguments:
        command -- Command to execute.
        params  -- Optional.  Dictionary of parameters (name-value pairs).
        kwargs  -- Optional keyword arguments (name=value pairs).

        Return:
        Data from command.

        """
        self._check_session()
        if not params:
            params = {}
        if kwargs:
            params.update(kwargs)
        params['command'] = command
        status, data = self._rest.post_request('perform', None, params)
        return data