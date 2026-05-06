def organization_memberships(self, state=None, number=-1, etag=None):
        """List organizations of which the user is a current or pending member.

        :param str state: (option), state of the membership, i.e., active,
            pending
        :returns: iterator of :class:`Membership <github3.orgs.Membership>`
        """
        params = None
        url = self._build_url('user', 'memberships', 'orgs')
        if state is not None and state.lower() in ('active', 'pending'):
            params = {'state': state.lower()}
        return self._iter(int(number), url, Membership,
                          params=params,
                          etag=etag)