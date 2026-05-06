def advance(self, id=None):
        """
        Advance to next token, optionally check that current token is 'id'
        """

        if id and self.token.id != id:
            raise SyntaxError("Expected {0}".format(id))
        self.token = self.next()