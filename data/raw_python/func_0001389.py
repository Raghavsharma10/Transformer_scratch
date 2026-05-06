def to_dict(self):
        """ Return a dict of the users. """
        users = dict(users=list())
        for user in self:
            users['users'].append(user.to_dict())
        return users