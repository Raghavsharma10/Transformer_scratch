def multi_send(self, template, emails, _vars=None, evars=None, schedule_time=None, options=None):
        """
        Remotely send an email template to multiple email addresses.
        http://docs.sailthru.com/api/send
        @param template: template string
        @param emails: List with email values or comma separated email string
        @param _vars: a key/value hash of the replacement vars to use in the send. Each var may be referenced as {varname} within the template itself
        @param options: optional dictionary to include replyto and/or test keys
        @param schedule_time: do not send the email immediately, but at some point in the future. Any date recognized by PHP's strtotime function is valid, but be sure to specify timezone or use a UTC time to avoid confusion
        """
        _vars = _vars or {}
        evars = evars or {}
        options = options or {}
        data = {'template': template,
                'email': ','.join(emails) if isinstance(emails, list) else emails,
                'vars': _vars.copy(),
                'evars': evars.copy(),
                'options': options.copy()}
        if schedule_time is not None:
            data['schedule_time'] = schedule_time
        return self.api_post('send', data)