def reserve_time_slot(self, id, cancel_existing=None, comments=None, participant_id=None):
        """
        Reserve a time slot.

        Reserves a particular time slot and return the new reservation
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # OPTIONAL - participant_id
        """User or group id for whom you are making the reservation (depends on the
        participant type). Defaults to the current user (or user's candidate group)."""
        if participant_id is not None:
            data["participant_id"] = participant_id

        # OPTIONAL - comments
        """Comments to associate with this reservation"""
        if comments is not None:
            data["comments"] = comments

        # OPTIONAL - cancel_existing
        """Defaults to false. If true, cancel any previous reservation(s) for this
        participant and appointment group."""
        if cancel_existing is not None:
            data["cancel_existing"] = cancel_existing

        self.logger.debug("POST /api/v1/calendar_events/{id}/reservations with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/calendar_events/{id}/reservations".format(**path), data=data, params=params, no_data=True)