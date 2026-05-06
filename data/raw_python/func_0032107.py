def _get_sender(self, sender):
        """
        Return a tuple of 3 elements
            0: the email signature "Me <email@me.com>"
            1: the name "Me"
            2: the email address "email@me.com"

        if sender is an email string, all 3 elements will be the email address
        """
        if isinstance(sender, tuple):
            return "%s <%s>" % sender, sender[0], sender[1]
        else:
            return sender, sender, sender