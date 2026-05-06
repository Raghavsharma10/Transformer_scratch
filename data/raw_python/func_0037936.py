def set_usage_rights_courses(self, file_ids, course_id, usage_rights_use_justification, folder_ids=None, publish=None, usage_rights_legal_copyright=None, usage_rights_license=None):
        """
        Set usage rights.

        Sets copyright and license information for one or more files
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - file_ids
        """List of ids of files to set usage rights for."""
        data["file_ids"] = file_ids

        # OPTIONAL - folder_ids
        """List of ids of folders to search for files to set usage rights for.
        Note that new files uploaded to these folders do not automatically inherit these rights."""
        if folder_ids is not None:
            data["folder_ids"] = folder_ids

        # OPTIONAL - publish
        """Whether the file(s) or folder(s) should be published on save, provided that usage rights have been specified (set to `true` to publish on save)."""
        if publish is not None:
            data["publish"] = publish

        # REQUIRED - usage_rights[use_justification]
        """The intellectual property justification for using the files in Canvas"""
        self._validate_enum(usage_rights_use_justification, ["own_copyright", "used_by_permission", "fair_use", "public_domain", "creative_commons"])
        data["usage_rights[use_justification]"] = usage_rights_use_justification

        # OPTIONAL - usage_rights[legal_copyright]
        """The legal copyright line for the files"""
        if usage_rights_legal_copyright is not None:
            data["usage_rights[legal_copyright]"] = usage_rights_legal_copyright

        # OPTIONAL - usage_rights[license]
        """The license that applies to the files. See the {api:UsageRightsController#licenses List licenses endpoint} for the supported license types."""
        if usage_rights_license is not None:
            data["usage_rights[license]"] = usage_rights_license

        self.logger.debug("PUT /api/v1/courses/{course_id}/usage_rights with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/courses/{course_id}/usage_rights".format(**path), data=data, params=params, single_item=True)