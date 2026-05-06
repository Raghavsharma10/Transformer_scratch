def get_user(self, userPk):
        """Returns the user specified with the user's Pk or UUID"""
        r = self._request('user/' + str(userPk))
        if r:
            # Set base properties and copy data inside the user
            u = User()
            u.pk = u.id = userPk
            u.__dict__.update(r.json())
            return u
        return None