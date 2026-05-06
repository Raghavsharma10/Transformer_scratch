def set_course_timetable(self, course_id, timetables_course_section_id=None, timetables_course_section_id_end_time=None, timetables_course_section_id_location_name=None, timetables_course_section_id_start_time=None, timetables_course_section_id_weekdays=None):
        """
        Set a course timetable.

        Creates and updates "timetable" events for a course.
        Can automaticaly generate a series of calendar events based on simple schedules
        (e.g. "Monday and Wednesday at 2:00pm" )
        
        Existing timetable events for the course and course sections
        will be updated if they still are part of the timetable.
        Otherwise, they will be deleted.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # OPTIONAL - timetables[course_section_id]
        """An array of timetable objects for the course section specified by course_section_id.
        If course_section_id is set to "all", events will be created for the entire course."""
        if timetables_course_section_id is not None:
            data["timetables[course_section_id]"] = timetables_course_section_id

        # OPTIONAL - timetables[course_section_id][weekdays]
        """A comma-separated list of abbreviated weekdays
        (Mon-Monday, Tue-Tuesday, Wed-Wednesday, Thu-Thursday, Fri-Friday, Sat-Saturday, Sun-Sunday)"""
        if timetables_course_section_id_weekdays is not None:
            data["timetables[course_section_id][weekdays]"] = timetables_course_section_id_weekdays

        # OPTIONAL - timetables[course_section_id][start_time]
        """Time to start each event at (e.g. "9:00 am")"""
        if timetables_course_section_id_start_time is not None:
            data["timetables[course_section_id][start_time]"] = timetables_course_section_id_start_time

        # OPTIONAL - timetables[course_section_id][end_time]
        """Time to end each event at (e.g. "9:00 am")"""
        if timetables_course_section_id_end_time is not None:
            data["timetables[course_section_id][end_time]"] = timetables_course_section_id_end_time

        # OPTIONAL - timetables[course_section_id][location_name]
        """A location name to set for each event"""
        if timetables_course_section_id_location_name is not None:
            data["timetables[course_section_id][location_name]"] = timetables_course_section_id_location_name

        self.logger.debug("POST /api/v1/courses/{course_id}/calendar_events/timetable with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/courses/{course_id}/calendar_events/timetable".format(**path), data=data, params=params, no_data=True)