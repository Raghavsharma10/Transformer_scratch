async def check_in(self):
        """ Checks this participant in

        |methcoro|

        Warning:
            |unstable|

        Raises:
            APIException

        """
        res = await self.connection('POST', 'tournaments/{}/participants/{}/check_in'.format(self._tournament_id, self._id))
        self._refresh_from_json(res)