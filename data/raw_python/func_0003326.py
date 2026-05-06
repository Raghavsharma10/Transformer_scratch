def basicauth(self, realm = b'all', nofail = False):
        "Try to get the basic authorize info, return (username, password) if succeeded, return 401 otherwise"
        if b'authorization' in self.headerdict:
            auth = self.headerdict[b'authorization']
            auth_pair = auth.split(b' ', 1)
            if len(auth_pair) < 2:
                raise HttpInputException('Authorization header is malformed')
            if auth_pair[0].lower() == b'basic':
                try:
                    userpass = base64.b64decode(auth_pair[1])
                except Exception:
                    raise HttpInputException('Invalid base-64 string')
                userpass_pair = userpass.split(b':', 1)
                if len(userpass_pair) != 2:
                    raise HttpInputException('Authorization header is malformed')
                return userpass_pair
        if nofail:
            return (None, None)
        else:
            self.basicauthfail(realm)