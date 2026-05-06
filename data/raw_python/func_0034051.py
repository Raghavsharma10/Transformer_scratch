def send_facebook(self, token):
        """
        Tells the server which Facebook account this client uses.

        After sending, the server takes some time to
        get the data from Facebook.

        Seems to be broken in recent versions of the game.
        """
        self.send_struct('<B%iB' % len(token), 81, *map(ord, token))
        self.facebook_token = token