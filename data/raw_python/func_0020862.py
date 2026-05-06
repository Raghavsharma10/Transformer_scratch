def send(self, template, email, _vars=None, options=None, schedule_time=None, limit=None):
        """
        Remotely send an email template to a single email address.
        http://docs.sailthru.com/api/send
        @param template: template string
        @param email: Email value
        @param _vars: a key/value hash of the replacement vars to use in the send. Each var may be referenced as {varname} within the template itself
        @param options: optional dictionary to include replyto and/or test keys
        @param limit: optional dictionary to name, time, and handle conflicts of limits
        @param schedule_time: do not send the email immediately, but at some point in the future. Any date recognized by PHP's strtotime function is valid, but be sure to specify timezone or use a UTC time to avoid confusion
        """
        _vars = _vars or {}
        options = options or {}
        data = {'template': template,
                'email': email,
                'vars': _vars,
                'options': options.copy()}
        if limit:
            data['limit'] = limit.copy()
        if schedule_time is not None:
            data['schedule_time'] = schedule_time
        return self.api_post('send', data)