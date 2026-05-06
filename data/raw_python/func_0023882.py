def profiles(self):
        '''
        return the rolls this people is related with
        '''
        limit = []

        if self.is_admin():
            limit.append(_("Administrator"))
        limit.sort()

        return limit