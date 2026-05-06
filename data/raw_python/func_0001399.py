def add_token_object(self, token):
        ''' Add a token object into this sentence '''
        token.sent = self  # take ownership of given token
        self.__tokens.append(token)
        return token