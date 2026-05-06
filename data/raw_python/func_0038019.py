def invite_others_to_group(self, group_id, invitees):
        """
        Invite others to a group.

        Sends an invitation to all supplied email addresses which will allow the
        receivers to join the group.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - group_id
        """ID"""
        path["group_id"] = group_id

        # REQUIRED - invitees
        """An array of email addresses to be sent invitations."""
        data["invitees"] = invitees

        self.logger.debug("POST /api/v1/groups/{group_id}/invite with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/groups/{group_id}/invite".format(**path), data=data, params=params, no_data=True)