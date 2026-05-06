def message_user(self, username, domain, subject, message):
        """Currently use send_message_chat and discard subject, because headline messages are not stored by
        mod_offline."""
        jid = '%s@%s' % (username, domain)
        if self.api_version <= (14, 7):
            # TODO: it's unclear when send_message was introduced
            command = 'send_message_chat'
            args = domain, '%s@%s' % (username, domain), message
        else:
            command = 'send_message'
            args = 'chat', domain, jid, subject, message

        code, out, err = self.ctl(command, *args)
        if code != 0:
            raise BackendError(code)