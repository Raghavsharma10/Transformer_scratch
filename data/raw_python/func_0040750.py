def send_message(self, target, content, uid=None):
        """
        Sends a message through IRC
        """
        # Compute maximum length of payload
        prefix = "PRIVMSG {0} :".format(target)
        single_prefix = self._make_line("MSG:")
        single_prefix_len = len(single_prefix)
        max_len = 510 - len(prefix)

        content_len = len(content)
        if (content_len + single_prefix_len) < max_len:
            # One pass message
            self._connection.send_raw("{0}{1}{2}" \
                                       .format(prefix, single_prefix, content))

        else:
            # Multiple-passes message
            uid = uid or str(uuid.uuid4()).replace('-', '').upper()
            prefix = "{0}{1}:".format(prefix, self._make_line(uid))
            max_len = 510 - len(prefix)

            self._connection.privmsg(target, self._make_line(uid, "BEGIN"))

            for chunk in chunks(content, max_len):
                self._connection.send_raw(''.join((prefix, chunk)))

            self._connection.privmsg(target, self._make_line(uid, "END"))