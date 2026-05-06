def pots(self, refresh=False):
        """
        Returns a list of pots owned by the currently authorised user.

        Official docs:
            https://monzo.com/docs/#pots

        :param refresh: decides if the pots information should be refreshed.
        :type refresh: bool
        :returns: list of Monzo pots
        :rtype: list of MonzoPot
        """
        if not refresh and self._cached_pots:
            return self._cached_pots

        endpoint = '/pots/listV1'
        response = self._get_response(
            method='get', endpoint=endpoint,
        )

        pots_json = response.json()['pots']
        pots = [MonzoPot(data=pot) for pot in pots_json]
        self._cached_pots = pots

        return pots