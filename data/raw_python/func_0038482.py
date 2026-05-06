def create_appointment_group(self, appointment_group_title, appointment_group_context_codes, appointment_group_description=None, appointment_group_location_address=None, appointment_group_location_name=None, appointment_group_max_appointments_per_participant=None, appointment_group_min_appointments_per_participant=None, appointment_group_new_appointments_X=None, appointment_group_participant_visibility=None, appointment_group_participants_per_appointment=None, appointment_group_publish=None, appointment_group_sub_context_codes=None):
        """
        Create an appointment group.

        Create and return a new appointment group. If new_appointments are
        specified, the response will return a new_appointments array (same format
        as appointments array, see "List appointment groups" action)
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - appointment_group[context_codes]
        """Array of context codes (courses, e.g. course_1) this group should be
        linked to (1 or more). Users in the course(s) with appropriate permissions
        will be able to sign up for this appointment group."""
        data["appointment_group[context_codes]"] = appointment_group_context_codes

        # OPTIONAL - appointment_group[sub_context_codes]
        """Array of sub context codes (course sections or a single group category)
        this group should be linked to. Used to limit the appointment group to
        particular sections. If a group category is specified, students will sign
        up in groups and the participant_type will be "Group" instead of "User"."""
        if appointment_group_sub_context_codes is not None:
            data["appointment_group[sub_context_codes]"] = appointment_group_sub_context_codes

        # REQUIRED - appointment_group[title]
        """Short title for the appointment group."""
        data["appointment_group[title]"] = appointment_group_title

        # OPTIONAL - appointment_group[description]
        """Longer text description of the appointment group."""
        if appointment_group_description is not None:
            data["appointment_group[description]"] = appointment_group_description

        # OPTIONAL - appointment_group[location_name]
        """Location name of the appointment group."""
        if appointment_group_location_name is not None:
            data["appointment_group[location_name]"] = appointment_group_location_name

        # OPTIONAL - appointment_group[location_address]
        """Location address."""
        if appointment_group_location_address is not None:
            data["appointment_group[location_address]"] = appointment_group_location_address

        # OPTIONAL - appointment_group[publish]
        """Indicates whether this appointment group should be published (i.e. made
        available for signup). Once published, an appointment group cannot be
        unpublished. Defaults to false."""
        if appointment_group_publish is not None:
            data["appointment_group[publish]"] = appointment_group_publish

        # OPTIONAL - appointment_group[participants_per_appointment]
        """Maximum number of participants that may register for each time slot.
        Defaults to null (no limit)."""
        if appointment_group_participants_per_appointment is not None:
            data["appointment_group[participants_per_appointment]"] = appointment_group_participants_per_appointment

        # OPTIONAL - appointment_group[min_appointments_per_participant]
        """Minimum number of time slots a user must register for. If not set, users
        do not need to sign up for any time slots."""
        if appointment_group_min_appointments_per_participant is not None:
            data["appointment_group[min_appointments_per_participant]"] = appointment_group_min_appointments_per_participant

        # OPTIONAL - appointment_group[max_appointments_per_participant]
        """Maximum number of time slots a user may register for."""
        if appointment_group_max_appointments_per_participant is not None:
            data["appointment_group[max_appointments_per_participant]"] = appointment_group_max_appointments_per_participant

        # OPTIONAL - appointment_group[new_appointments][X]
        """Nested array of start time/end time pairs indicating time slots for this
        appointment group. Refer to the example request."""
        if appointment_group_new_appointments_X is not None:
            data["appointment_group[new_appointments][X]"] = appointment_group_new_appointments_X

        # OPTIONAL - appointment_group[participant_visibility]
        """"private":: participants cannot see who has signed up for a particular
                    time slot
        "protected":: participants can see who has signed up.  Defaults to
                      "private"."""
        if appointment_group_participant_visibility is not None:
            self._validate_enum(appointment_group_participant_visibility, ["private", "protected"])
            data["appointment_group[participant_visibility]"] = appointment_group_participant_visibility

        self.logger.debug("POST /api/v1/appointment_groups with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/appointment_groups".format(**path), data=data, params=params, no_data=True)