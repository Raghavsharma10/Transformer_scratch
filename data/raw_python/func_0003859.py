def auth_required(self):
        """
        If any ancestor required an authentication, this node needs it too.
        """
        if self._auth:
            return self._auth, self
        return self.__parent__.auth_required()