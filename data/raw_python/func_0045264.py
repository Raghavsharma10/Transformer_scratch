async def update_notifications(self, on_match_open: bool = None, on_tournament_end: bool = None):
        """ update participants notifications for this tournament

        |methcoro|

        Args:
            on_match_open: Email registered Challonge participants when matches open up for them
            on_tournament_end: Email registered Challonge participants the results when this tournament ends

        Raises:
            APIException

        """
        params = {}
        if on_match_open is not None:
            params['notify_users_when_matches_open'] = on_match_open
        if on_tournament_end is not None:
            params['notify_users_when_the_tournament_ends'] = on_tournament_end
        assert_or_raise(len(params) > 0, ValueError, 'At least one of the notifications must be given')
        await self.update(**params)