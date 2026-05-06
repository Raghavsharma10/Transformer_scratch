async def create_tournament(self, name: str, url: str, tournament_type: TournamentType = TournamentType.single_elimination, **params) -> Tournament:
        """ creates a simple tournament with basic options

        |methcoro|

        Args:
            name: name of the new tournament
            url: url of the new tournament (http://challonge.com/url)
            tournament_type: Defaults to TournamentType.single_elimination
            params: optional params (see http://api.challonge.com/v1/documents/tournaments/create)

        Returns:
            Tournament: the newly created tournament

        Raises:
            APIException

        """
        params.update({
            'name': name,
            'url': url,
            'tournament_type': tournament_type.value,
        })
        res = await self.connection('POST', 'tournaments', 'tournament', **params)
        self._refresh_tournament_from_json(res)
        return self._find_tournament_by_id(res['tournament']['id'])