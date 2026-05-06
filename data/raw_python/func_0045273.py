async def process_check_ins(self):
        """ finalize the check in phase

        |methcoro|

        Warning:
            |unstable|

        Note:
            |from_api| This should be invoked after a tournament's check-in window closes before the tournament is started.
            1. Marks participants who have not checked in as inactive.
            2. Moves inactive participants to bottom seeds (ordered by original seed).
            3. Transitions the tournament state from 'checking_in' to 'checked_in'
            NOTE: Checked in participants on the waiting list will be promoted if slots become available.

        Raises:
            APIException

        """
        params = {
                'include_participants': 1,  # forced to 1 since we need to update the Participant instances
                'include_matches': 1 if AUTO_GET_MATCHES else 0
            }
        res = await self.connection('POST', 'tournaments/{}/process_check_ins'.format(self._id), **params)
        self._refresh_from_json(res)