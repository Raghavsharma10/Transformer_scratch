def delete_calendar_event(self, id, cancel_reason=None):
        """
        Delete a calendar event.

        Delete an event from the calendar and return the deleted event
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # OPTIONAL - cancel_reason
        """Reason for deleting/canceling the event."""
        if cancel_reason is not None:
            params["cancel_reason"] = cancel_reason

        self.logger.debug("DELETE /api/v1/calendar_events/{id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("DELETE", "/api/v1/calendar_events/{id}".format(**path), data=data, params=params, no_data=True)