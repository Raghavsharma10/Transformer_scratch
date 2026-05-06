def get_export_configuration(self, config_id):
        """
        Retrieve the ExportConfiguration with the given ID

        :param string config_id:
            ID for which to search
        :return:
            a :class:`meteorpi_model.ExportConfiguration` or None, or no match was found.
        """
        sql = (
            'SELECT uid, exportConfigId, exportType, searchString, targetURL, '
            'targetUser, targetPassword, exportName, description, active '
            'FROM archive_exportConfig WHERE exportConfigId = %s')
        return first_from_generator(
                self.generators.export_configuration_generator(sql=sql, sql_args=(config_id,)))