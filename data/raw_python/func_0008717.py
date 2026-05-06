def get_thread(self):
        """Return the message thread this Message is in."""
        url = (self._imgur._base_url + "/3/message/{0}/thread".format(
               self.first_message.id))
        resp = self._imgur._send_request(url)
        return [Message(msg, self._imgur) for msg in resp]