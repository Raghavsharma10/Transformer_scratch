def cmd_echo(self, connection, sender, target, payload):
        """
        Echoes the given payload
        """
        connection.privmsg(target, payload or "Hello, {0}".format(sender))