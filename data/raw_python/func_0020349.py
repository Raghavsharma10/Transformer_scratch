def assign(self, role):
        '''Assign :class:`Role` ``role`` to this :class:`Subject`. If this
:class:`Subject` is the :attr:`Role.owner`, this method does nothing.'''
        if role.owner_id != self.id:
            return self.roles.add(role)