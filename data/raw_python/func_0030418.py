def passwordLogin(self, username):
        """
        Generate a new challenge for the given username.
        """
        self.challenge = secureRandom(16)
        self.username = username
        return {'challenge': self.challenge}