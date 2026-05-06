async def undo_check_in(self):
        """ Undo the check in for this participant

        |methcoro|

        Warning:
            |unstable|

        Raises:
            APIException

        """
        res = await self.connection('POST', 'tournaments/{}/participants/{}/undo_check_in'.format(self._tournament_id, self._id))
        self._refresh_from_json(res)