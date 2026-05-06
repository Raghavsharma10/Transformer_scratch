def get_notifications(self, new=True):
        """Return all the notifications for this user."""
        url = (self._imgur._base_url + "/3/account/{0}/"
               "notifications".format(self.name))
        resp = self._imgur._send_request(url, params=locals(), needs_auth=True)
        msgs = [Message(msg_dict, self._imgur, has_fetched=True) for msg_dict
                in resp['messages']]
        replies = [Comment(com_dict, self._imgur, has_fetched=True) for
                   com_dict in resp['replies']]
        return {'messages': msgs, 'replies': replies}