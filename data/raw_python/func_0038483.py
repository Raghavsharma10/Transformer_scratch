def get_next_appointment(self, appointment_group_ids=None):
        """
        Get next appointment.

        Return the next appointment available to sign up for. The appointment
        is returned in a one-element array. If no future appointments are
        available, an empty array is returned.
        """
        path = {}
        data = {}
        params = {}

        # OPTIONAL - appointment_group_ids
        """List of ids of appointment groups to search."""
        if appointment_group_ids is not None:
            params["appointment_group_ids"] = appointment_group_ids

        self.logger.debug("GET /api/v1/appointment_groups/next_appointment with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/appointment_groups/next_appointment".format(**path), data=data, params=params, all_pages=True)