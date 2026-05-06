async def shuffle_participants(self):
        """ Shuffle participants' seeds

        |methcoro|

        Note:
            |from_api| Randomize seeds among participants. Only applicable before a tournament has started.

        Raises:
            APIException

        """
        res = await self.connection('POST', 'tournaments/{}/participants/randomize'.format(self._id))
        self._refresh_participants_from_json(res)