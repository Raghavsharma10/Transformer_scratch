def message_user(self, username, domain, subject, message):
        """Currently use send_message_chat and discard subject, because headline messages are not
        stored by mod_offline."""

        kwargs = {
            'body': message,
            'from': domain,
            'to': '%s@%s' % (username, domain),
        }

        if self.api_version <= (14, 7):
            # TODO: it's unclear when send_message was introduced
            command = 'send_message_chat'
        else:
            command = 'send_message'
            kwargs['subject'] = subject
            kwargs['type'] = 'normal'
        result = self.rpc(command, **kwargs)

        if result['res'] == 0:
            return
        else:
            raise BackendError(result.get('text', 'Unknown Error'))