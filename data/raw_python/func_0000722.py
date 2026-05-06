def send(self, envelope):
        """
        Send an *envelope* which may be an envelope
        or an enclosure-like object, see
        :class:`~mailthon.enclosure.Enclosure` and
        :class:`~mailthon.envelope.Envelope`, and
        returns a :class:`~mailthon.response.SendmailResponse`
        object.
        """
        rejected = self.conn.sendmail(
            stringify_address(envelope.sender),
            [stringify_address(k) for k in envelope.receivers],
            envelope.string(),
        )
        status_code, reason = self.conn.noop()
        return SendmailResponse(
            status_code,
            reason,
            rejected,
        )