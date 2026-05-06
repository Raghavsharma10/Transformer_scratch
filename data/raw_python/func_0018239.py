def authentication_ok(self):
        '''Checks the authentication and sets the username of the currently connected terminal.  Returns True or False'''
        username = None
        password = None
        if self.authCallback:
            if self.authNeedUser:
                username = self.readline(prompt=self.PROMPT_USER, use_history=False)
            if self.authNeedPass:
                password = self.readline(echo=False, prompt=self.PROMPT_PASS, use_history=False)
                if self.DOECHO:
                    self.write("\n")
            try:
                self.authCallback(username, password)
            except:
                self.username = None
                return False
            else:
                # Successful authentication
                self.username = username
                return True
        else:
            # No authentication desired
            self.username = None
            return True