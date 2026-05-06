def update_calendar_event(self, id, calendar_event_child_event_data_X_context_code=None, calendar_event_child_event_data_X_end_at=None, calendar_event_child_event_data_X_start_at=None, calendar_event_context_code=None, calendar_event_description=None, calendar_event_end_at=None, calendar_event_location_address=None, calendar_event_location_name=None, calendar_event_start_at=None, calendar_event_time_zone_edited=None, calendar_event_title=None):
        """
        Update a calendar event.

        Update and return a calendar event
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # OPTIONAL - calendar_event[context_code]
        """Context code of the course/group/user to move this event to.
        Scheduler appointments and events with section-specific times cannot be moved between calendars."""
        if calendar_event_context_code is not None:
            data["calendar_event[context_code]"] = calendar_event_context_code

        # OPTIONAL - calendar_event[title]
        """Short title for the calendar event."""
        if calendar_event_title is not None:
            data["calendar_event[title]"] = calendar_event_title

        # OPTIONAL - calendar_event[description]
        """Longer HTML description of the event."""
        if calendar_event_description is not None:
            data["calendar_event[description]"] = calendar_event_description

        # OPTIONAL - calendar_event[start_at]
        """Start date/time of the event."""
        if calendar_event_start_at is not None:
            data["calendar_event[start_at]"] = calendar_event_start_at

        # OPTIONAL - calendar_event[end_at]
        """End date/time of the event."""
        if calendar_event_end_at is not None:
            data["calendar_event[end_at]"] = calendar_event_end_at

        # OPTIONAL - calendar_event[location_name]
        """Location name of the event."""
        if calendar_event_location_name is not None:
            data["calendar_event[location_name]"] = calendar_event_location_name

        # OPTIONAL - calendar_event[location_address]
        """Location address"""
        if calendar_event_location_address is not None:
            data["calendar_event[location_address]"] = calendar_event_location_address

        # OPTIONAL - calendar_event[time_zone_edited]
        """Time zone of the user editing the event. Allowed time zones are
        {http://www.iana.org/time-zones IANA time zones} or friendlier
        {http://api.rubyonrails.org/classes/ActiveSupport/TimeZone.html Ruby on Rails time zones}."""
        if calendar_event_time_zone_edited is not None:
            data["calendar_event[time_zone_edited]"] = calendar_event_time_zone_edited

        # OPTIONAL - calendar_event[child_event_data][X][start_at]
        """Section-level start time(s) if this is a course event. X can be any
        identifier, provided that it is consistent across the start_at, end_at
        and context_code"""
        if calendar_event_child_event_data_X_start_at is not None:
            data["calendar_event[child_event_data][X][start_at]"] = calendar_event_child_event_data_X_start_at

        # OPTIONAL - calendar_event[child_event_data][X][end_at]
        """Section-level end time(s) if this is a course event."""
        if calendar_event_child_event_data_X_end_at is not None:
            data["calendar_event[child_event_data][X][end_at]"] = calendar_event_child_event_data_X_end_at

        # OPTIONAL - calendar_event[child_event_data][X][context_code]
        """Context code(s) corresponding to the section-level start and end time(s)."""
        if calendar_event_child_event_data_X_context_code is not None:
            data["calendar_event[child_event_data][X][context_code]"] = calendar_event_child_event_data_X_context_code

        self.logger.debug("PUT /api/v1/calendar_events/{id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/calendar_events/{id}".format(**path), data=data, params=params, no_data=True)