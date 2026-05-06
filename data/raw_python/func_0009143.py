def fail(self, tup):
        """Indicate that processing of a Tuple has failed.

        :param tup: the Tuple to fail (its ``id`` if ``str``).
        :type tup: :class:`str` or :class:`pystorm.component.Tuple`
        """
        tup_id = tup.id if isinstance(tup, Tuple) else tup
        self.send_message({"command": "fail", "id": tup_id})