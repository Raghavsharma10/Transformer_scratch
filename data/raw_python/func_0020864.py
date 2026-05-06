def set_email(self, email, _vars=None, lists=None, templates=None, verified=0, optout=None, send=None, send_vars=None):
        """
        DEPRECATED!
        Update information about one of your users, including adding and removing the user from lists.
        http://docs.sailthru.com/api/email
        """
        _vars = _vars or {}
        lists = lists or []
        templates = templates or []
        send_vars = send_vars or []
        data = {'email': email,
                'vars':  _vars.copy(),
                'lists': lists,
                'templates': templates,
                'verified': int(verified)}
        if optout is not None:
            data['optout'] = optout
        if send is not None:
            data['send'] = send
        if send_vars:
            data['send_vars'] = send_vars
        return self.api_post('email', data)