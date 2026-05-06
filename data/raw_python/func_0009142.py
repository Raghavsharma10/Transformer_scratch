def ack(self, tup):
        """Indicate that processing of a Tuple has succeeded.

        :param tup: the Tuple to acknowledge.
        :type tup: :class:`str` or :class:`pystorm.component.Tuple`
        """
        tup_id = tup.id if isinstance(tup, Tuple) else tup
        self.send_message({"command": "ack", "id": tup_id})