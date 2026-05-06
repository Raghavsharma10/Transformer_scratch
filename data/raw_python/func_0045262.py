async def update(self, **params):
        """ update some parameters of the tournament

        Use this function if you want to update multiple options at once, but prefer helpers functions like :func:`allow_attachments`, :func:`set_start_date`...

        |methcoro|

        Args:
            params: one or more of: ``name`` ``tournament_type`` ``url`` ``subdomain`` ``description`` ``open_signup``
                                    ``hold_third_place_match`` ``pts_for_match_win`` ``pts_for_match_tie`` ``pts_for_game_win``
                                    ``pts_for_game_tie`` ``pts_for_bye`` ``swiss_rounds`` ``ranked_by`` ``rr_pts_for_match_win``
                                    ``rr_pts_for_match_tie`` ``rr_pts_for_game_win`` ``rr_pts_for_game_tie`` ``accept_attachments``
                                    ``hide_forum`` ``show_rounds`` ``private`` ``notify_users_when_matches_open``
                                    ``notify_users_when_the_tournament_ends`` ``sequential_pairings`` ``signup_cap``
                                    ``start_at`` ``check_in_duration`` ``grand_finals_modifier``

        Raises:
            APIException

        """
        assert_or_raise(all(k in self._update_parameters for k in params.keys()),
                        NameError,
                        'Wrong parameter given')

        res = await self.connection('PUT',
                                    'tournaments/{}'.format(self._id),
                                    'tournament',
                                    **params)
        self._refresh_from_json(res)