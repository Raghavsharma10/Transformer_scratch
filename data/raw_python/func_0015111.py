def success(self):
        """
        Checks for the presence of errors in the response. Returns ``True`` if
        all is well, ``False`` otherwise.

        :rtype: bool
        :returns ``True`` if PayPal says our query was successful.
        """
        return self.ack.upper() in (self.config.ACK_SUCCESS,
                                    self.config.ACK_SUCCESS_WITH_WARNING)