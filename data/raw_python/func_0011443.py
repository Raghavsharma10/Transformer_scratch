def restart_user(self, subid):
    '''restart a user, which means revoking and issuing a new token.'''
    p = self.revoke_token(subid)
    p = self.refresh_token(subid)
    return p