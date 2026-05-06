def members(self):
        '''Member components if this component is composite.'''
        with self._mutex:
            if not self._members:
                self._members = {}
                for o in self.organisations:
                    # TODO: Search for these in the tree
                    self._members[o.org_id] = o.obj.get_members()
        return self._members