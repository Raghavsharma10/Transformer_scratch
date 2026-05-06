async def reopen(self):
        """ Reopens a match that was marked completed, automatically resetting matches that follow it

        |methcoro|

        Raises:
            APIException

        """
        res = await self.connection('POST', 'tournaments/{}/matches/{}/reopen'.format(self._tournament_id, self._id))
        self._refresh_from_json(res)