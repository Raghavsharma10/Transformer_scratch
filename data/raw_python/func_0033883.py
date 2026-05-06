def send_mail(self, sender, subject, recipients, message, response_id=None, html_message=False):
        """Send an email using EBUio features. If response_id is set, replies will be send back to the PlugIt server."""

        params = {
            'sender': sender,
            'subject': subject,
            'dests': recipients,
            'message': message,
            'html_message': html_message,
        }

        if response_id:
            params['response_id'] = response_id

        return self._request('mail/', postParams=params, verb='POST')