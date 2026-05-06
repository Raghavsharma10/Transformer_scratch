def create_or_update_export_configuration(self, export_config):
        """
        Create a new file export configuration or update an existing one

        :param ExportConfiguration export_config:
            a :class:`meteorpi_model.ExportConfiguration` containing the specification for the export. If this
            doesn't include a 'config_id' field it will be inserted as a new record in the database and the field will
            be populated, updating the supplied object. If it does exist already this will update the other properties
            in the database to match the supplied object.
        :returns:
            The supplied :class:`meteorpi_model.ExportConfiguration` as stored in the DB. This is guaranteed to have
            its 'config_id' string field defined.
        """
        search_string = json.dumps(obj=export_config.search.as_dict())
        user_id = export_config.user_id
        password = export_config.password
        target_url = export_config.target_url
        enabled = export_config.enabled
        name = export_config.name
        description = export_config.description
        export_type = export_config.type
        if export_config.config_id is not None:
            # Update existing record
            self.con.execute(
                    'UPDATE archive_exportConfig c '
                    'SET c.searchString = %s, c.targetUrl = %s, c.targetUser = %s, c.targetPassword = %s, '
                    'c.exportName = %s, c.description = %s, c.active = %s, c.exportType = %s '
                    'WHERE c.exportConfigId = %s',
                    (search_string, target_url, user_id, password, name, description, enabled, export_type,
                     export_config.config_id))
        else:
            # Create new record and add the ID into the supplied config
            item_id = mp.get_hash(mp.now(), name, export_type)
            self.con.execute(
                    'INSERT INTO archive_exportConfig '
                    '(searchString, targetUrl, targetUser, targetPassword, '
                    'exportName, description, active, exportType, exportConfigId) '
                    'VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ',
                    (search_string, target_url, user_id, password,
                     name, description, enabled, export_type, item_id))
            export_config.config_id = item_id
        return export_config