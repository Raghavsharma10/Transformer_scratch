def reserve(self, force=False):
        """ Reserve port.

        :param force: True - take forcefully, False - fail if port is reserved by other user
        """

        if not force:
            try:
                self.api.call_rc('ixPortTakeOwnership {}'.format(self.uri))
            except Exception as _:
                raise TgnError('Failed to take ownership for port {} current owner is {}'.format(self, self.owner))
        else:
            self.api.call_rc('ixPortTakeOwnership {} force'.format(self.uri))