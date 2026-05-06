def create_or_update_events_directly_for_course_timetable(self, course_id, course_section_id=None, events=None, events_code=None, events_end_at=None, events_location_name=None, events_start_at=None):
        """
        Create or update events directly for a course timetable.

        Creates and updates "timetable" events for a course or course section.
        Similar to {api:CalendarEventsApiController#set_course_timetable setting a course timetable},
        but instead of generating a list of events based on a timetable schedule,
        this endpoint expects a complete list of events.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # OPTIONAL - course_section_id
        """Events will be created for the course section specified by course_section_id.
        If not present, events will be created for the entire course."""
        if course_section_id is not None:
            data["course_section_id"] = course_section_id

        # OPTIONAL - events
        """An array of event objects to use."""
        if events is not None:
            data["events"] = events

        # OPTIONAL - events[start_at]
        """Start time for the event"""
        if events_start_at is not None:
            data["events[start_at]"] = events_start_at

        # OPTIONAL - events[end_at]
        """End time for the event"""
        if events_end_at is not None:
            data["events[end_at]"] = events_end_at

        # OPTIONAL - events[location_name]
        """Location name for the event"""
        if events_location_name is not None:
            data["events[location_name]"] = events_location_name

        # OPTIONAL - events[code]
        """A unique identifier that can be used to update the event at a later time
        If one is not specified, an identifier will be generated based on the start and end times"""
        if events_code is not None:
            data["events[code]"] = events_code

        self.logger.debug("POST /api/v1/courses/{course_id}/calendar_events/timetable_events with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/courses/{course_id}/calendar_events/timetable_events".format(**path), data=data, params=params, no_data=True)