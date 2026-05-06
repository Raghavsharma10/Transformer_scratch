def import_sis_data(self, account_id, add_sis_stickiness=None, attachment=None, batch_mode=None, batch_mode_term_id=None, clear_sis_stickiness=None, diffing_data_set_identifier=None, diffing_remaster_data_set=None, extension=None, import_type=None, override_sis_stickiness=None):
        """
        Import SIS data.

        Import SIS data into Canvas. Must be on a root account with SIS imports
        enabled.

        For more information on the format that's expected here, please see the
        "SIS CSV" section in the API docs.
        """
        path = {}
        data = {}
        params = {}
        files = {}

        # REQUIRED - PATH - account_id
        """ID"""
        path["account_id"] = account_id

        # OPTIONAL - import_type
        """Choose the data format for reading SIS data. With a standard Canvas
        install, this option can only be 'instructure_csv', and if unprovided,
        will be assumed to be so. Can be part of the query string."""
        if import_type is not None:
            data["import_type"] = import_type

        # OPTIONAL - attachment
        """There are two ways to post SIS import data - either via a
        multipart/form-data form-field-style attachment, or via a non-multipart
        raw post request.

        'attachment' is required for multipart/form-data style posts. Assumed to
        be SIS data from a file upload form field named 'attachment'.

        Examples:
          curl -F attachment=@<filename> -H "Authorization: Bearer <token>" \
              'https://<canvas>/api/v1/accounts/<account_id>/sis_imports.json?import_type=instructure_csv'

        If you decide to do a raw post, you can skip the 'attachment' argument,
        but you will then be required to provide a suitable Content-Type header.
        You are encouraged to also provide the 'extension' argument.

        Examples:
          curl -H 'Content-Type: application/octet-stream' --data-binary @<filename>.zip \
              -H "Authorization: Bearer <token>" \
              'https://<canvas>/api/v1/accounts/<account_id>/sis_imports.json?import_type=instructure_csv&extension=zip'

          curl -H 'Content-Type: application/zip' --data-binary @<filename>.zip \
              -H "Authorization: Bearer <token>" \
              'https://<canvas>/api/v1/accounts/<account_id>/sis_imports.json?import_type=instructure_csv'

          curl -H 'Content-Type: text/csv' --data-binary @<filename>.csv \
              -H "Authorization: Bearer <token>" \
              'https://<canvas>/api/v1/accounts/<account_id>/sis_imports.json?import_type=instructure_csv'

          curl -H 'Content-Type: text/csv' --data-binary @<filename>.csv \
              -H "Authorization: Bearer <token>" \
              'https://<canvas>/api/v1/accounts/<account_id>/sis_imports.json?import_type=instructure_csv&batch_mode=1&batch_mode_term_id=15'"""
        if attachment is not None:
            if type(attachment) is file:
                files['attachment'] = (os.path.basename(attachment.name), attachment)
            elif os.path.exists(attachment):
                files['attachment'] = (os.path.basename(attachment), open(attachment, 'rb'))
            else:
                raise ValueError('The attachment must be an open file or a path to a readable file.')

        # OPTIONAL - extension
        """Recommended for raw post request style imports. This field will be used to
        distinguish between zip, xml, csv, and other file format extensions that
        would usually be provided with the filename in the multipart post request
        scenario. If not provided, this value will be inferred from the
        Content-Type, falling back to zip-file format if all else fails."""
        if extension is not None:
            data["extension"] = extension

        # OPTIONAL - batch_mode
        """If set, this SIS import will be run in batch mode, deleting any data
        previously imported via SIS that is not present in this latest import.
        See the SIS CSV Format page for details."""
        if batch_mode is not None:
            data["batch_mode"] = batch_mode

        # OPTIONAL - batch_mode_term_id
        """Limit deletions to only this term. Required if batch mode is enabled."""
        if batch_mode_term_id is not None:
            data["batch_mode_term_id"] = batch_mode_term_id

        # OPTIONAL - override_sis_stickiness
        """Many fields on records in Canvas can be marked "sticky," which means that
        when something changes in the UI apart from the SIS, that field gets
        "stuck." In this way, by default, SIS imports do not override UI changes.
        If this field is present, however, it will tell the SIS import to ignore
        "stickiness" and override all fields."""
        if override_sis_stickiness is not None:
            data["override_sis_stickiness"] = override_sis_stickiness

        # OPTIONAL - add_sis_stickiness
        """This option, if present, will process all changes as if they were UI
        changes. This means that "stickiness" will be added to changed fields.
        This option is only processed if 'override_sis_stickiness' is also provided."""
        if add_sis_stickiness is not None:
            data["add_sis_stickiness"] = add_sis_stickiness

        # OPTIONAL - clear_sis_stickiness
        """This option, if present, will clear "stickiness" from all fields touched
        by this import. Requires that 'override_sis_stickiness' is also provided.
        If 'add_sis_stickiness' is also provided, 'clear_sis_stickiness' will
        overrule the behavior of 'add_sis_stickiness'"""
        if clear_sis_stickiness is not None:
            data["clear_sis_stickiness"] = clear_sis_stickiness

        # OPTIONAL - diffing_data_set_identifier
        """If set on a CSV import, Canvas will attempt to optimize the SIS import by
        comparing this set of CSVs to the previous set that has the same data set
        identifier, and only appliying the difference between the two. See the
        SIS CSV Format documentation for more details."""
        if diffing_data_set_identifier is not None:
            data["diffing_data_set_identifier"] = diffing_data_set_identifier

        # OPTIONAL - diffing_remaster_data_set
        """If true, and diffing_data_set_identifier is sent, this SIS import will be
        part of the data set, but diffing will not be performed. See the SIS CSV
        Format documentation for details."""
        if diffing_remaster_data_set is not None:
            data["diffing_remaster_data_set"] = diffing_remaster_data_set

        self.logger.debug("POST /api/v1/accounts/{account_id}/sis_imports with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/accounts/{account_id}/sis_imports".format(**path), data=data, params=params, files=files, single_item=True)