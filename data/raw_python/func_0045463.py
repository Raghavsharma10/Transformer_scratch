async def get_tournaments(self, subdomain: str = None, force_update: bool = False) -> list:
        """ gets all user's tournaments

        |methcoro|

        Args:
            subdomain: *optional* subdomain needs to be given explicitely to get tournaments in a subdomain
            force_update: *optional* set to True to force the data update from Challonge

        Returns:
            list[Tournament]: list of all the user tournaments

        Raises:
            APIException

        """
        if self.tournaments is None:
            force_update = True
            self._subdomains_searched.append('' if subdomain is None else subdomain)
        elif subdomain is None and '' not in self._subdomains_searched:
            force_update = True
            self._subdomains_searched.append('')
        elif subdomain is not None and subdomain not in self._subdomains_searched:
            force_update = True
            self._subdomains_searched.append(subdomain)

        if force_update:
            params = {
                'include_participants': 1 if AUTO_GET_PARTICIPANTS else 0,
                'include_matches': 1 if AUTO_GET_MATCHES else 0
            }
            if subdomain is not None:
                params['subdomain'] = subdomain

            res = await self.connection('GET', 'tournaments', **params)
            if len(res) == 0:
                self.tournaments = []
            else:
                for t_data in res:
                    self._refresh_tournament_from_json(t_data)

        return self.tournaments