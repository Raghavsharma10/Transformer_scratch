async def reset(self):
        """ reset the tournament on Challonge

        |methcoro|

        Note:
            |from_api| Reset a tournament, clearing all of its scores and attachments. You can then add/remove/edit participants before starting the tournament again.

        Raises:
            APIException

        """
        params = {
                'include_participants': 1 if AUTO_GET_PARTICIPANTS else 0,
                'include_matches': 1 if AUTO_GET_MATCHES else 0
            }
        res = await self.connection('POST', 'tournaments/{}/reset'.format(self._id), **params)
        self._refresh_from_json(res)