def list_members_of_collaboration(self, id, include=None):
        """
        List members of a collaboration.

        List the collaborators of a given collaboration
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # OPTIONAL - include
        """- "collaborator_lti_id": Optional information to include with each member.
          Represents an identifier to be used for the member in an LTI context.
        - "avatar_image_url": Optional information to include with each member.
          The url for the avatar of a collaborator with type 'user'."""
        if include is not None:
            self._validate_enum(include, ["collaborator_lti_id", "avatar_image_url"])
            params["include"] = include

        self.logger.debug("GET /api/v1/collaborations/{id}/members with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/collaborations/{id}/members".format(**path), data=data, params=params, all_pages=True)