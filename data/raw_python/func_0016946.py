def edit(self, state):
        """Edit the user's membership.

        :param str state: (required), the state the membership should be in.
            Only accepts ``"active"``.
        :returns: itself
        """
        if state and state.lower() == 'active':
            data = dumps({'state': state.lower()})
            json = self._json(self._patch(self._api, data=data))
            self._update_attributes(json)
        return self