def get_message(self, id):
        """
        Return a Message object for given id.

        :param id: The id of the message object to return.
        """
        url = self._base_url + "/3/message/{0}".format(id)
        resp = self._send_request(url)
        return Message(resp, self)