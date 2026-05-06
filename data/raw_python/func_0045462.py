async def get_tournament(self, t_id: int = None, url: str = None, subdomain: str = None, force_update=False) -> Tournament:
        """ gets a tournament with its id or url or url+subdomain
        Note: from the API, it can't be known if the retrieved tournament was made from this user.
        Thus, any tournament  is added to the local list of tournaments, but some functions (updates/destroy...) cannot be used for tournaments not owned by this user.

        |methcoro|

        Args:
            t_id: tournament id
            url: last part of the tournament url (http://challonge.com/XXX)
            subdomain: first part of the tournament url, if any (http://XXX.challonge.com/...)
            force_update: *optional* set to True to force the data update from Challonge

        Returns:
            Tournament

        Raises:
            APIException
            ValueError: if neither of the arguments are provided

        """
        assert_or_raise((t_id is None) ^ (url is None),
                        ValueError,
                        'One of t_id or url must not be None')

        found_t = self._find_tournament_by_id(t_id) if t_id is not None else self._find_tournament_by_url(url, subdomain)
        if force_update or found_t is None:
            param = t_id
            if param is None:
                if subdomain is not None:
                    param = '{}-{}'.format(subdomain, url)
                else:
                    param = url
            res = await self.connection('GET', 'tournaments/{}'.format(param))
            self._refresh_tournament_from_json(res)
            found_t = self._find_tournament_by_id(res['tournament']['id'])

        return found_t