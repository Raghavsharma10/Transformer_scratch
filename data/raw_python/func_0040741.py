def safe_send(self, connection, target, message, *args, **kwargs):
        """
        Safely sends a message to the given target
        """
        # Compute maximum length of payload
        prefix = "PRIVMSG {0} :".format(target)
        max_len = 510 - len(prefix)

        for chunk in chunks(message.format(*args, **kwargs), max_len):
            connection.send_raw("{0}{1}".format(prefix, chunk))