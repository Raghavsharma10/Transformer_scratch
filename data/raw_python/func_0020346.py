def check_user(self, username, email):
        '''username and email (if provided) must be unique.'''
        users = self.router.user
        avail = yield users.filter(username=username).count()
        if avail:
            raise FieldError('Username %s not available' % username)
        if email:
            avail = yield users.filter(email=email).count()
            if avail:
                raise FieldError('Email %s not available' % email)