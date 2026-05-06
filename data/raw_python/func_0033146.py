def share(self, with_user, roles=None):
        '''
        Give cube access rights to another user
        '''
        return self.proxy.share(table=self.name, with_user=with_user,
                                roles=roles)